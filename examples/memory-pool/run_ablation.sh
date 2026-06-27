#!/usr/bin/env bash
# run_ablation.sh — HeteroPool memory-pool ablation experiment runner.
#
# Two calling conventions:
#   (a) Full matrix mode (no args):
#         ./run_ablation.sh [-- python args ...]
#       Delegates to run_ablation.py to run 4 modes × 3 scenarios × 10 reps.
#
#   (b) Single-run mode (mode + yaml args):
#         ./run_ablation.sh <full|nodma|nofpview|alloff> <yaml>
#       Runs one dataflow with the given ablation settings.  Useful for
#       ad-hoc testing during development.
#
# Examples:
#   ./run_ablation.sh                          # full matrix
#   ./run_ablation.sh -n 2                     # quick smoke test
#   ./run_ablation.sh --dry-run                # preview plan
#   ./run_ablation.sh nodma cpu2cuda.yml       # single ad-hoc run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ------------------------------------------------------------------
# Full-matrix mode (no args, or args starting with '-')
# ------------------------------------------------------------------
if [ $# -eq 0 ] || [ "${1:0:1}" = "-" ]; then
    exec python3 "$SCRIPT_DIR/run_ablation.py" "$@"
fi

# ------------------------------------------------------------------
# Single-run mode (mode + yaml)
# ------------------------------------------------------------------
MODE="${1:?Usage: $0 <full|nodma|nofpview|alloff> <yaml>}"
YAML="${2:?Usage: $0 <full|nodma|nofpview|alloff> <yaml>}"

case "$MODE" in
  full)     ;;
  nodma)    export HETEROPOOL_NO_DMA=1 ;;
  nofpview) export HETEROPOOL_NO_DMA=1
             export HETEROPOOL_NO_FASTPATH_VIEW=1 ;;
  alloff)   export HETEROPOOL_NO_DMA=1
             export HETEROPOOL_NO_FASTPATH_VIEW=1
             export HETEROPOOL_NO_POOL_REUSE=1 ;;
  *)
    echo "Unknown mode '$MODE'. Valid: full, nodma, nofpview, alloff"
    exit 1 ;;
esac

echo "=== Ablation single run ==="
echo "Mode:  $MODE"
echo "YAML:  $YAML"
echo "Env:"
env | grep -E 'HETEROPOOL_' || echo "  (none)"
echo "==========================="

dora run "$YAML" --stop-after 100s
