#!/usr/bin/env python3
"""HeteroPool memory-pool ablation experiment runner.

Drives 4 experiment modes × 3 device-pair scenarios × N repetitions,
collecting throughput metrics and producing statistical summaries.

Usage:
  python3 run_ablation.py                  # full matrix, 10 reps
  python3 run_ablation.py -n 2             # quick smoke test
  python3 run_ablation.py --skip-gpu       # CPU-only scenarios
  python3 run_ablation.py --dry-run        # print plan, don't run
  python3 run_ablation.py --rebuild        # rebuild before running
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, stdev
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEMORY_POOL_DIR = Path(__file__).resolve().parent
DORA_BIN = shutil.which("dora") or str(PROJECT_ROOT / "target" / "release" / "dora")

# Throughput regex — matches receiver.py output line
THROUGHPUT_RE = re.compile(
    r"Average transfer throughput:\s+([0-9]+(?:\.[0-9]+)?)\s*MB/s"
)

# Experiment modes: label → env vars dict.
# Per-dimension independent baselines: each mode disables ONE optimization while
# the other two remain at their optimal configuration for the data size.
#   Full:      pinned DMA + fast-path view + pool reuse   (optimal baseline)
#   Pageable:  pageable DMA (no cudaHostRegister)         (DMA path ablation)
#   NoFastPath: daemon slow-path every frame               (fast-path ablation)
#   NoReuse:   fresh pool per frame (no pool reuse)        (pool-reuse ablation)
MODES: dict[str, dict[str, str]] = {
    "full":       {},
    "pageable":   {"HETEROPOOL_NO_PIN": "1"},
    "nofastpath": {"HETEROPOOL_NO_FASTPATH": "1"},
    "noreuse":    {"HETEROPOOL_NO_REUSE": "1"},
}

# Scenarios: label → (yaml_filename, needs_gpu)
SCENARIOS: dict[str, tuple[str, bool]] = {
    "cpu2cpu": ("cpu2cpu.yml", False),
    "cpu2cuda": ("cpu2cuda.yml", True),
    "cuda2cpu": ("cuda2cpu.yml", True),
}

DEFAULT_REPETITIONS = 10
DEFAULT_TIMEOUT = 120  # seconds per run

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    mode: str
    scenario: str
    run: int
    throughput_mbs: Optional[float] = None
    status: str = "SUCCESS"  # SUCCESS | FAILED | TIMEOUT | SKIPPED
    duration_s: float = 0.0
    notes: str = ""


@dataclass
class ExperimentPlan:
    modes: list[str] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    repetitions: int = DEFAULT_REPETITIONS
    total_runs: int = 0
    gpu_available: bool = False
    build_fresh: bool = True


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------


class AblationRunner:
    def __init__(
        self,
        repetitions: int = DEFAULT_REPETITIONS,
        timeout: int = DEFAULT_TIMEOUT,
        skip_gpu: bool = False,
        skip_build_check: bool = False,
        rebuild: bool = False,
        dry_run: bool = False,
        output_dir: Optional[str] = None,
        verbose: bool = False,
        tensor_bytes: int = 0,
    ):
        self.repetitions = repetitions
        self.timeout = timeout
        self.skip_gpu = skip_gpu
        self.skip_build_check = skip_build_check
        self.rebuild = rebuild
        self.dry_run = dry_run
        self.verbose = verbose
        self.tensor_bytes = tensor_bytes
        self.results: list[RunResult] = []
        self.start_time: float = 0.0
        self.output_dir: Optional[Path] = None
        self._user_output_dir = output_dir
        self._interrupted = False

        # Install SIGINT handler for graceful shutdown
        signal.signal(signal.SIGINT, self._on_sigint)

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def detect_gpu(self) -> bool:
        """Check whether CUDA is available via PyTorch."""
        try:
            import torch  # noqa: F401

            return torch.cuda.is_available()
        except ImportError:
            return False

    def check_build(self) -> bool:
        """Return True if the Python extension .so is newer than lib.rs."""
        lib_rs = PROJECT_ROOT / "apis" / "python" / "node" / "src" / "lib.rs"
        so_files = list(
            (PROJECT_ROOT / "target" / "release").glob("libdora_node_api_python*.so")
        )
        if not so_files:
            return False
        newest_so = max(f.stat().st_mtime for f in so_files)
        return newest_so >= lib_rs.stat().st_mtime

    def rebuild_project(self) -> bool:
        """Run cargo build for the Python extension and CLI."""
        print("=== Rebuilding dora (Python extension + CLI) ===")
        rc = subprocess.run(
            ["cargo", "build", "--release", "-p", "dora-node-api-python", "-p", "dora-cli"],
            cwd=PROJECT_ROOT,
        ).returncode
        if rc != 0:
            print("FATAL: cargo build failed")
            return False
        print("  Build: OK\n")
        return True

    def cleanup_stale(self) -> None:
        """Kill orphaned dora processes and clean stale state.

        Stale lock files in out/ and .dora/python-envs/ survive crashes and
        can cause the next ``dora run`` to block waiting for a lock held by a
        process that no longer exists.  Removing them between runs prevents
        the experiment from hanging.
        """
        for name in ("dora-daemon", "dora-coordinator"):
            subprocess.run(
                ["pkill", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # Clean leftover shared-memory files
        for pattern in ("dora_pool_*", "dora_shm_*"):
            for f in Path("/dev/shm").glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass
        # Clean stale dora session lock files (survive crashes / SIGKILL)
        for lock in MEMORY_POOL_DIR.glob("out/*.lock"):
            try:
                lock.unlink()
            except OSError:
                pass
        # Clean stale per-node Python-env locks
        for lock in MEMORY_POOL_DIR.glob(".dora/python-envs/*/.lock"):
            try:
                lock.unlink()
            except OSError:
                pass
        time.sleep(0.5)

    def gpu_memory_used(self) -> Optional[int]:
        """Return GPU memory used in MiB, or None if nvidia-smi unavailable."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            return int(out.strip().split("\n")[0])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Single-run execution
    # ------------------------------------------------------------------

    def run_single(
        self, mode: str, scenario: str, run_num: int, log_file: Path
    ) -> RunResult:
        """Execute one dora run and return the result."""
        scenario_yaml, needs_gpu = SCENARIOS[scenario]
        yaml_path = MEMORY_POOL_DIR / scenario_yaml
        env_vars = MODES[mode]

        result = RunResult(mode=mode, scenario=scenario, run=run_num)
        t0 = time.perf_counter()

        # Build environment
        run_env = os.environ.copy()
        run_env.update(env_vars)
        if self.tensor_bytes:
            run_env["TENSOR_BYTES"] = str(self.tensor_bytes)

        stop_after_s = max(15, self.timeout - 10)
        cmd = [DORA_BIN, "run", str(yaml_path), "--stop-after", f"{stop_after_s}s"]

        if self.dry_run:
            env_str = " ".join(f"{k}={v}" for k, v in env_vars.items()) if env_vars else "(none)"
            print(
                f"  [{mode}] {scenario} run {run_num}/{self.repetitions}  "
                f"env=({env_str})  cmd={' '.join(cmd)}"
            )
            result.status = "SKIPPED"
            result.notes = "dry-run"
            return result

        try:
            proc = subprocess.run(
                cmd,
                cwd=MEMORY_POOL_DIR,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            result.duration_s = time.perf_counter() - t0

            # Write logs
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                f.write(f"# command: {' '.join(cmd)}\n")
                f.write(f"# env: {env_vars}\n")
                f.write(f"# exit_code: {proc.returncode}\n")
                f.write(f"# duration_s: {result.duration_s:.1f}\n")
                f.write("# stdout:\n")
                f.write(proc.stdout)
                if proc.stderr:
                    f.write("\n# stderr:\n")
                    f.write(proc.stderr)

            # Check for Python traceback in output (indicates node crash)
            if "Traceback (most recent call last)" in proc.stdout or \
               "Traceback (most recent call last)" in proc.stderr:
                result.status = "FAILED"
                result.notes = "Python traceback detected in output"
                return result

            if proc.returncode != 0:
                result.status = "FAILED"
                result.notes = f"exit code {proc.returncode}"
                return result

            # Parse throughput
            match = THROUGHPUT_RE.search(proc.stdout)
            if match:
                result.throughput_mbs = float(match.group(1))
            else:
                result.status = "FAILED"
                result.notes = "throughput line not found in output"

        except subprocess.TimeoutExpired:
            result.duration_s = time.perf_counter() - t0
            result.status = "TIMEOUT"
            result.notes = f"timed out after {self.timeout}s"
            # Write whatever we captured
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                f.write(f"# TIMEOUT after {self.timeout}s\n")

        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_all(self, plan: ExperimentPlan) -> bool:
        """Execute the full experiment matrix. Returns True if all runs completed."""
        if plan.total_runs == 0:
            print("No experiments to run (all scenarios skipped?).")
            return False

        # Set up output directory
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        self.output_dir = (
            Path(self._user_output_dir)
            if self._user_output_dir
            else MEMORY_POOL_DIR / "results" / f"ablation_{ts}"
        )
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Results directory: {self.output_dir}")
        print()

        if self.dry_run:
            print("=== DRY RUN — no experiments will execute ===\n")

        self.start_time = time.perf_counter()
        total = plan.total_runs
        completed = 0
        run_index = 0

        for mode in plan.modes:
            for scenario in plan.scenarios:
                _, needs_gpu = SCENARIOS[scenario]
                for r in range(1, self.repetitions + 1):
                    run_index += 1

                    if self._interrupted:
                        print("\nInterrupted — saving partial results...")
                        self._save_results()
                        return False

                    # Cleanup (skip first run)
                    if run_index > 1:
                        self.cleanup_stale()

                    # GPU memory check before GPU runs
                    gpu_before = None
                    if needs_gpu and not self.dry_run:
                        gpu_before = self.gpu_memory_used()

                    # Execute
                    log_name = f"{mode}__{scenario}__run{r:02d}.log"
                    log_path = logs_dir / log_name
                    result = self.run_single(mode, scenario, r, log_path)
                    self.results.append(result)

                    # GPU memory check after
                    if needs_gpu and not self.dry_run and result.status == "SUCCESS":
                        gpu_after = self.gpu_memory_used()
                        if gpu_before is not None and gpu_after is not None:
                            delta = gpu_after - gpu_before
                            if delta > 100:
                                print(
                                    f"  WARNING: GPU memory leak {delta} MiB "
                                    f"({gpu_before} → {gpu_after})"
                                )

                    # Progress
                    completed += 1
                    elapsed = time.perf_counter() - self.start_time
                    if completed > 0:
                        eta = (elapsed / completed) * (total - completed)
                    else:
                        eta = 0

                    status_mark = {
                        "SUCCESS": "OK",
                        "FAILED": "FAIL",
                        "TIMEOUT": "TIME",
                        "SKIPPED": "SKIP",
                    }.get(result.status, result.status)

                    detail = ""
                    if result.throughput_mbs is not None:
                        detail = f"  {result.throughput_mbs:.1f} MB/s  {result.duration_s:.1f}s"
                    elif result.notes:
                        detail = f"  {result.notes}"

                    print(
                        f"  [{completed:3d}/{total}] [{status_mark:4s}] "
                        f"[{mode}] {scenario} run {r}/{self.repetitions}{detail}"
                    )

        # Final cleanup
        self.cleanup_stale()

        elapsed_total = time.perf_counter() - self.start_time
        print(f"\nAll {total} runs completed in {elapsed_total:.0f}s")
        self._save_results()
        self._print_summary()
        return True

    # ------------------------------------------------------------------
    # Results & reporting
    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        """Write raw_data.csv, summary.csv, and report.md."""
        if not self.output_dir or not self.results:
            return

        # raw_data.csv
        raw_csv = self.output_dir / "raw_data.csv"
        with open(raw_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["mode", "scenario", "run", "throughput_mbs", "status", "duration_s", "notes"]
            )
            for r in self.results:
                writer.writerow(
                    [
                        r.mode,
                        r.scenario,
                        r.run,
                        f"{r.throughput_mbs:.1f}" if r.throughput_mbs is not None else "",
                        r.status,
                        f"{r.duration_s:.1f}",
                        r.notes,
                    ]
                )

        # Group results
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in self.results:
            if r.status == "SUCCESS" and r.throughput_mbs is not None:
                groups[(r.mode, r.scenario)].append(r.throughput_mbs)

        # summary.csv
        summary_csv = self.output_dir / "summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["mode", "scenario", "success", "failed", "median_mbs",
                 "min_mbs", "max_mbs", "stddev_mbs"]
            )
            for (mode, scenario), values in sorted(groups.items()):
                n_failed = sum(
                    1
                    for r in self.results
                    if r.mode == mode and r.scenario == scenario and r.status != "SUCCESS"
                )
                if len(values) >= 2:
                    writer.writerow(
                        [
                            mode,
                            scenario,
                            len(values),
                            n_failed,
                            f"{median(values):.1f}",
                            f"{min(values):.1f}",
                            f"{max(values):.1f}",
                            f"{stdev(values):.1f}",
                        ]
                    )
                elif len(values) == 1:
                    writer.writerow(
                        [
                            mode,
                            scenario,
                            1,
                            n_failed,
                            f"{values[0]:.1f}",
                            f"{values[0]:.1f}",
                            f"{values[0]:.1f}",
                            "N/A",
                        ]
                    )
                else:
                    writer.writerow(
                        [mode, scenario, 0, n_failed, "N/A", "N/A", "N/A", "N/A"]
                    )

        # report.md
        report_md = self.output_dir / "report.md"
        with open(report_md, "w") as f:
            f.write("# HeteroPool Ablation Experiment Report\n\n")
            f.write(
                f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            )
            f.write(f"**Total runs:** {len(self.results)}\n\n")

            f.write("## Results Summary\n\n")
            f.write(
                "| Mode | Scenario | Success | Median (MB/s) | Min | Max | StdDev |\n"
            )
            f.write(
                "|------|----------|---------|--------------|-----|-----|--------|\n"
            )
            for (mode, scenario), values in sorted(groups.items()):
                n_success = len(values)
                n_failed = sum(
                    1
                    for r in self.results
                    if r.mode == mode
                    and r.scenario == scenario
                    and r.status != "SUCCESS"
                )
                if n_success > 0:
                    std_str = f"{stdev(values):.1f}" if n_success >= 2 else "N/A"
                    f.write(
                        f"| {mode} | {scenario} | {n_success} | "
                        f"{median(values):.1f} | {min(values):.1f} | "
                        f"{max(values):.1f} | "
                        f"{std_str} |\n"
                    )
                else:
                    f.write(
                        f"| {mode} | {scenario} | 0/{n_success + n_failed} | "
                        f"N/A | N/A | N/A | N/A |\n"
                    )
            f.write("\n")

            # Failed runs
            failed = [r for r in self.results if r.status not in ("SUCCESS", "SKIPPED")]
            if failed:
                f.write("## Failed Runs\n\n")
                f.write("| Mode | Scenario | Run | Status | Notes |\n")
                f.write("|------|----------|-----|--------|-------|\n")
                for r in failed:
                    f.write(
                        f"| {r.mode} | {r.scenario} | {r.run} | "
                        f"{r.status} | {r.notes} |\n"
                    )
                f.write("\n")

            f.write("## Data Files\n\n")
            f.write("- [raw_data.csv](raw_data.csv) — per-run throughput values\n")
            f.write("- [summary.csv](summary.csv) — per-group statistics\n")
            f.write("- [logs/](logs/) — full stdout/stderr for each run\n")

        print(f"  Results saved to {self.output_dir}")

    def _print_summary(self) -> None:
        """Print a compact summary table to stdout."""
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in self.results:
            if r.status == "SUCCESS" and r.throughput_mbs is not None:
                groups[(r.mode, r.scenario)].append(r.throughput_mbs)

        if not groups:
            print("\nNo successful runs — no summary available.")
            return

        print("\n=== Ablation Summary (median MB/s) ===\n")
        scenarios = sorted({s for _, s in groups})
        modes = sorted({m for m, _ in groups})

        # Header
        header = f"{'Mode':<12}" + "".join(f"{s:>14}" for s in scenarios)
        print(header)
        print("-" * len(header))

        for mode in modes:
            row = f"{mode:<12}"
            for scenario in scenarios:
                values = groups.get((mode, scenario), [])
                if values:
                    row += f"{median(values):>14.1f}"
                else:
                    row += f"{'N/A':>14}"
            print(row)

        print()

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _on_sigint(self, signum, frame):
        self._interrupted = True
        print("\nInterrupt received — finishing current run, then saving...")


# ---------------------------------------------------------------------------
# Plan printer (used by --dry-run and startup)
# ---------------------------------------------------------------------------


def print_plan(plan: ExperimentPlan) -> None:
    print("=== HeteroPool Ablation Experiment Plan ===\n")
    print(f"  Modes:       {', '.join(plan.modes)}")
    print(f"  Scenarios:   {', '.join(plan.scenarios)}")
    print(f"  Repetitions: {plan.repetitions}")
    print(f"  Total runs:  {plan.total_runs}")
    print(f"  GPU:         {'available' if plan.gpu_available else 'NOT available'}")
    print(f"  Build:       {'fresh' if plan.build_fresh else 'STALE — consider --rebuild'}")
    print()

    est_seconds = plan.total_runs * 15  # rough estimate: 15s per run average
    print(f"  Estimated wall time: ~{est_seconds // 60}m {est_seconds % 60}s")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HeteroPool memory-pool ablation experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      full matrix (4 modes × 3 scenarios × 10 reps)
  %(prog)s -n 2                 quick smoke test (2 reps)
  %(prog)s --skip-gpu           CPU-only scenarios
  %(prog)s --dry-run            print plan without executing
  %(prog)s --rebuild            rebuild before running
  %(prog)s -o results/my_run    custom output directory
        """,
    )
    parser.add_argument(
        "-n", "--repetitions", type=int, default=DEFAULT_REPETITIONS,
        help=f"Repetitions per experiment (default: {DEFAULT_REPETITIONS})",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-run timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip scenarios that require CUDA GPU",
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Skip build freshness check",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild (cargo build --release) before running",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print experiment plan without executing",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Override default results directory",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show dora stdout/stderr in real time",
    )
    parser.add_argument(
        "-s", "--tensor-bytes", type=int, default=0,
        help="Tensor size in bytes (default: 15000*512*8 = 61.44 MB). "
             "Use small sizes (64K–1M) to make ablation overheads visible; "
             "at large sizes (>10 MB) data-copy bandwidth dominates.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build the experiment plan
    # ------------------------------------------------------------------
    runner = AblationRunner(
        repetitions=args.repetitions,
        timeout=args.timeout,
        skip_gpu=args.skip_gpu,
        skip_build_check=args.no_build,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        verbose=args.verbose,
        tensor_bytes=args.tensor_bytes,
    )

    gpu_available = runner.detect_gpu()
    build_fresh = runner.check_build() if not args.no_build else True

    active_modes = list(MODES.keys())
    active_scenarios = [
        s for s, (_, needs_gpu) in SCENARIOS.items()
        if not needs_gpu or (gpu_available and not args.skip_gpu)
    ]

    plan = ExperimentPlan(
        modes=active_modes,
        scenarios=active_scenarios,
        repetitions=args.repetitions,
        total_runs=len(active_modes) * len(active_scenarios) * args.repetitions,
        gpu_available=gpu_available,
        build_fresh=build_fresh,
    )

    print_plan(plan)

    # Pre-flight checks
    if not args.dry_run:
        if not build_fresh:
            print(
                "WARNING: Python extension .so is older than lib.rs.\n"
                "  Rust changes may not take effect. Use --rebuild to rebuild.\n"
            )
        if args.rebuild:
            if not runner.rebuild_project():
                sys.exit(1)
        if not args.no_build and not args.rebuild:
            # Re-check after potential warning
            pass

        if not Path(DORA_BIN).exists():
            print(f"FATAL: dora binary not found at {DORA_BIN}")
            sys.exit(1)

        # Pre-cleanup
        runner.cleanup_stale()

    # Run
    success = runner.run_all(plan)
    if not success and not args.dry_run:
        print("\nExperiment suite did not complete — partial results saved.")


if __name__ == "__main__":
    main()
