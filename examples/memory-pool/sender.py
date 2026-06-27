#!/usr/bin/env python
"""Send tensors through the memory-pool example dataflow."""

import os
import time

import numpy as np
import pyarrow as pa
import torch
from dora import Node
from dora.cuda import get_tensor_info

SIZE = 15000 * 512
# TENSOR_BYTES overrides SIZE when set (env var for future size-variation experiments).
# SIZE is in int64 elements, TENSOR_BYTES is in bytes.
TENSOR_BYTES = int(os.getenv("TENSOR_BYTES", "0"))
if TENSOR_BYTES > 0:
    SIZE = TENSOR_BYTES // 8  # int64 = 8 bytes per element
MESSAGE_COUNT = int(os.getenv("message_num", "100"))
SENDER_DEVICE = os.getenv("sender_device", "cpu")
RECEIVER_DEVICE = os.getenv("receiver_device", "cpu")
SCENARIO = os.getenv("memory_pool_scenario", "throughput")

NO_REUSE = os.environ.get("HETEROPOOL_NO_REUSE") == "1"
# Ablation: when HETEROPOOL_NO_PIN=1, write_memory_pool skips
# cudaHostRegister/Unregister on the source tensor, so cudaMemcpy
# uses pageable memory (Pageable mode).
NO_PIN = os.environ.get("HETEROPOOL_NO_PIN", "0") == "1"

node = Node("sender_node")
data_generation = np.random.default_rng()

memory_pool_id = None
for i in range(MESSAGE_COUNT):
    random_data = data_generation.integers(1000, size=SIZE, dtype=np.int64)
    random_data[0] = i  # monotonic counter lets receiver detect change without collision risk
    torch_tensor = torch.tensor(random_data, dtype=torch.int64, device=SENDER_DEVICE)
    t_send = time.perf_counter_ns()
    metadata = {"t_send": t_send, "scenario": SCENARIO}

    tensor_info = get_tensor_info(torch_tensor)

    if NO_REUSE:
        # Ablation: register a fresh pool every frame, write once.
        # The receiver reads the pool, validates, and sends next_require.
        # After the receiver is done (node.next() returns below), we
        # free the pool from the sender side to clean up GPU buffers
        # and pinned host memory that were allocated in this process
        # (cudaMalloc / cudaHostRegister — per-process resources that
        # the receiver's free_memory_pool cannot reach).
        memory_pool_id = node.register_memory_pool(tensor_info, RECEIVER_DEVICE)
        if i == 0:
            print(f"Sender preview: {torch_tensor[:5]}")
        node.write_memory_pool(memory_pool_id, tensor_info, if_pinned=not NO_PIN)
        node.send_output("data", memory_pool_id, metadata)
    elif i == 0:
        print(f"Sender preview: {torch_tensor[:5]}")
        memory_pool_id = node.register_memory_pool(tensor_info, RECEIVER_DEVICE)
        node.send_output("data", memory_pool_id, metadata)
    else:
        if SCENARIO == "write_after_free" and i == 1:
            node.free_memory_pool(memory_pool_id)
        node.write_memory_pool(memory_pool_id, tensor_info, if_pinned=not NO_PIN)
        node.send_output("data", pa.array([]), metadata)

    node.next()
    if NO_REUSE:
        # Receiver has read the pool and sent next_require.  Now clean up
        # the sender-side resources (GPU buffer via cudaMalloc, pinned
        # host memory via cudaHostRegister) — these live in the sender
        # process and the receiver's free_memory_pool cannot reach them.
        # The daemon-side deregistration is a no-op if the receiver already
        # freed it (warn_missing_memory_pool is graceful).
        node.free_memory_pool(memory_pool_id)
