# NVSHMEM4Py Examples: Annotated Walkthroughs

> **Purpose**: This document provides detailed, annotated walkthroughs of NVSHMEM Python examples, tracing the complete call path from Python user code through Cython bindings, C++ wrappers, down to device CUDA kernels and back.
>
> **Format**: For each example, we provide:
> - Code overview and purpose
> - Line-by-line annotated Python source with inline comments
> - Complete call stack from Python → Cython → C++ → CUDA
> - Visual diagrams of data flow
> - Links to implementation details across language boundaries

---

## Table of Contents

1. [Python → C++ Bridge Overview](#python--c-bridge-overview)
2. [Example 1: init_fini.py - Initialization Methods](#example-1-init_finipy---initialization-methods)
3. [Example 2: on-stream.py - Stream-Based Collectives](#example-2-on-streampy---stream-based-collectives)
4. [Example 3: simple_p2p_kernel.py - Basic RMA](#example-3-simple_p2p_kernelpy---basic-rma)
5. [Example 4: device_custom_kernel.py - Numba Device Code](#example-4-device_custom_kernelpy---numba-device-code)
6. [Numba Integration Deep Dive](#numba-integration-deep-dive)

---

## Python → C++ Bridge Overview

### Architecture Layers

```
┌────────────────────────────────────────────────────────────────┐
│          NVSHMEM4Py: Python to C++ Call Path                   │
└────────────────────────────────────────────────────────────────┘

Layer 1: PYTHON USER CODE
  │ nvshmem.core.reduce(team, dest, src, "sum", stream=stream)
  │
  ▼
Layer 2: PYTHON CORE API (Pure Python)
  │ [nvshmem4py/nvshmem/core/collective.py:79+]
  │ - Type checking
  │ - Device management
  │ - Error handling
  │
  ▼
Layer 3: CYTHON BINDINGS (Python ↔ C++)
  │ [nvshmem4py/nvshmem/bindings/cynvshmem.pyx:175+]
  │ cdef int nvshmemx_float_sum_reduce_on_stream(...)
  │     return _nvshmem._nvshmemx_float_sum_reduce_on_stream(...)
  │
  ▼
Layer 4: CYTHON INTERNAL (_internal/)
  │ [nvshmem4py/nvshmem/bindings/_internal/nvshmem.pyx]
  │ cdef extern from "nvshmemx.h":
  │     int _nvshmemx_float_sum_reduce_on_stream(...) nogil
  │
  ▼
Layer 5: C++ HOST API
  │ [src/include/host/nvshmemx_coll_api.h:500+]
  │ int nvshmemx_float_sum_reduce_on_stream(...)
  │
  ▼
Layer 6: C++ IMPLEMENTATION
  │ [src/host/coll/rdxn/rdxn_on_stream.cpp:100+]
  │ - Algorithm selection
  │ - Kernel launch
  │
  ▼
Layer 7: CUDA DEVICE KERNEL
  │ [src/include/non_abi/device/coll/reducescatter.cuh:50+]
  │ __global__ void reduce_kernel(...)
  │ - GPU execution
  │
  ▼
Layer 8: HARDWARE (GPU, NVLink, InfiniBand)
```

### Key Files in Python Bridge

| Component | Location | Purpose |
|-----------|----------|---------|
| **User API** | `nvshmem4py/nvshmem/core/*.py` | Pure Python interface |
| **Cython Wrappers** | `nvshmem4py/nvshmem/bindings/cynvshmem.pyx` | Python→C++ bridge |
| **Cython Internal** | `nvshmem4py/nvshmem/bindings/_internal/nvshmem.pyx` | C++ declarations |
| **Numba Bindings** | `nvshmem4py/nvshmem/bindings/device/numba/` | Device-side Python |
| **Core Modules** | `nvshmem4py/nvshmem/core/` | Memory, collectives, RMA |

---

## Example 1: init_fini.py - Initialization Methods

### Purpose

Demonstrates **multiple initialization methods**:
- MPI-based init
- UID-based init with MPI broadcast
- UID-based init with torchrun
- Emulated MPI init

### Source Code with Annotations: MPI UID Init

```python
# nvshmem4py/examples/init_fini.py

"""
This file contains examples of initialization and finalization of NVSHMEM
through various launching methods
"""
import numpy as np
import nvshmem.core
from cuda.core.experimental import Device, system
import os

from mpi4py import MPI
import torch
import torch.distributed as dist

# [1] ═══════════════════════════════════════════════════════════
# EXAMPLE 1: MPI UID-based Initialization
# ═══════════════════════════════════════════════════════════════
def mpi_uid_init():
    from mpi4py import MPI

    # [2] Get MPI context
    comm = MPI.COMM_WORLD              # LINE 16
    rank = comm.Get_rank()             # LINE 17
    nranks = comm.Get_size()           # LINE 18
    # CALL STACK:
    #   └─> MPI.COMM_WORLD [mpi4py wrapper]
    #        └─> MPI_Comm_rank() / MPI_Comm_size() [MPI C library]

    # [3] ═══════════════════════════════════════════════════════════
    # SET CUDA DEVICE
    # ═══════════════════════════════════════════════════════════════
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices  # LINE 20
    dev = Device(local_rank_per_node)  # LINE 21
    dev.set_current()                  # LINE 22
    # CALL STACK:
    #   └─> cuda.core.Device(device_id)  [cuda-python]
    #        └─> cudaSetDevice(device_id) [CUDA runtime]

    # [4] ═══════════════════════════════════════════════════════════
    # CREATE UNIQUE ID (ROOT ONLY)
    # ═══════════════════════════════════════════════════════════════
    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)  # LINE 25
    # CALL STACK:
    #   └─> nvshmem.core.get_unique_id() [nvshmem4py/nvshmem/core/init_fini.py:79+]
    #        │
    #        ├─> load_nvidia_dynamic_lib("nvshmem_host")  # Load shared library
    #        └─> bindings.uniqueid()  # Create UniqueID object
    #             │
    #             └─> [nvshmem4py/nvshmem/bindings/cynvshmem.pyx]
    #                  Creates struct wrapper for nvshmemx_uniqueid_t

    if rank == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()  # LINE 28
        # CALL STACK (for rank 0):
        #   └─> bindings.get_uniqueid(unique_id.ptr)
        #        │
        #        └─> [nvshmem4py/nvshmem/bindings/_internal/nvshmem.pyx]
        #             cdef extern from "nvshmemx.h":
        #                 int _nvshmemx_get_uniqueid(nvshmemx_uniqueid_t *uid)
        #             │
        #             └─> nvshmemx_get_uniqueid() [src/host/init/init.cu:267+]
        #                  │
        #                  ├─> bootstrap_preinit(NVSHMEMX_INIT_WITH_UNIQUEID, ...)
        #                  │    └─> Initialize bootstrap (UID mode)
        #                  │
        #                  └─> boot_handle.pre_init_ops->get_unique_id(uid)
        #                       │
        #                       └─> [src/modules/bootstrap/uid/bootstrap_uid.cpp:100+]
        #                            ├─> Generate random 128-bit UUID
        #                            └─> Store in uid->internal array

    # [5] ═══════════════════════════════════════════════════════════
    # BROADCAST UID TO ALL RANKS
    # ═══════════════════════════════════════════════════════════════
    # Broadcast UID to all ranks
    comm.Bcast(uniqueid._data.view(np.int8), root=0)  # LINE 31
    # CALL STACK:
    #   └─> mpi4py.MPI.Comm.Bcast() [mpi4py wrapper]
    #        └─> MPI_Bcast(buffer, count, MPI_INT8, root, comm) [MPI C library]
    #
    # uniqueid._data is a NumPy structured array:
    #   dtype = np.dtype([('internal', np.uint8, 128)])
    #   Corresponds to nvshmemx_uniqueid_t.internal[128]

    # [6] ═══════════════════════════════════════════════════════════
    # INITIALIZE NVSHMEM WITH UID
    # ═══════════════════════════════════════════════════════════════
    nvshmem.core.init(
        device=dev,
        uid=uniqueid,
        rank=rank,
        nranks=nranks,
        mpi_comm=None,
        initializer_method="uid"
    )  # LINE 33-34
    # CALL STACK:
    #   └─> nvshmem.core.init() [nvshmem4py/nvshmem/core/init_fini.py:121+]
    #        │
    #        ├─> Validation:
    #        │   ├─> Check device is cuda.core.Device
    #        │   ├─> Check uid is bindings.uniqueid
    #        │   ├─> Check rank/nranks are valid
    #        │   └─> Check initializer_method == "uid"
    #        │
    #        ├─> Load NVSHMEM library:
    #        │   └─> load_nvidia_dynamic_lib("nvshmem_host")
    #        │
    #        ├─> Build init attributes:
    #        │   └─> bindings.InitAttr()
    #        │        ├─> Set bootstrap flags: NVSHMEMX_INIT_WITH_UNIQUEID
    #        │        └─> bindings.set_attr_uniqueid_args(rank, nranks, uid, attr)
    #        │             │
    #        │             └─> [nvshmem4py/nvshmem/bindings/cynvshmem.pyx:800+]
    #        │                  └─> _nvshmem._nvshmemx_set_attr_uniqueid_args(...)
    #        │                       │
    #        │                       └─> nvshmemx_set_attr_uniqueid_args()
    #        │                            [src/host/init/init.cu:290+]
    #        │                            Stores rank, nranks, uid in attr struct
    #        │
    #        ├─> Call C++ init:
    #        │   └─> bindings.init_thread(NVSHMEM_THREAD_SERIALIZED, attr)
    #        │        │
    #        │        └─> [nvshmem4py/nvshmem/bindings/_internal/nvshmem.pyx]
    #        │             cdef int _nvshmemi_init_thread(...)
    #        │             │
    #        │             └─> nvshmemi_init_thread() [src/host/init/init.cu:500+]
    #        │                  │
    #        │                  ├─> Parse environment variables
    #        │                  ├─> Bootstrap init (UID mode):
    #        │                  │    └─> bootstrap_init(NVSHMEMX_INIT_WITH_UNIQUEID, attr)
    #        │                  │         │
    #        │                  │         └─> [src/modules/bootstrap/uid/bootstrap_uid.cpp:200+]
    #        │                  │              ├─> Extract rank, nranks from attr
    #        │                  │              ├─> Create TCP sockets for coordination
    #        │                  │              ├─> Exchange endpoint info
    #        │                  │              └─> Build PE connectivity map
    #        │                  │
    #        │                  ├─> Topology discovery
    #        │                  ├─> Symmetric heap allocation
    #        │                  ├─> Transport initialization
    #        │                  ├─> Team creation
    #        │                  └─> Device state update
    #        │
    #        └─> Python bookkeeping:
    #             ├─> Set _is_initialized["status"] = INITIALIZED
    #             ├─> Cache device: _cached_device = dev
    #             └─> Register finalize cleanup

    # [7] ═══════════════════════════════════════════════════════════
    # FINALIZE
    # ═══════════════════════════════════════════════════════════════
    nvshmem.core.finalize()  # LINE 35
    # CALL STACK:
    #   └─> nvshmem.core.finalize() [nvshmem4py/nvshmem/core/init_fini.py:250+]
    #        │
    #        ├─> memory._free_all_buffers()  # Free any leaked symmetric memory
    #        │    └─> For each buffer in _mr_references:
    #        │         └─> nvshmem_free(ptr)
    #        │
    #        ├─> bindings.finalize()
    #        │    │
    #        │    └─> nvshmemi_finalize() [src/host/init/init.cu:1200+]
    #        │         ├─> Close transports
    #        │         ├─> Free symmetric heap
    #        │         ├─> Destroy teams
    #        │         └─> Shutdown bootstrap
    #        │
    #        └─> Cleanup Python state:
    #             ├─> _is_initialized["status"] = NOT_INITIALIZED
    #             └─> Clear _cached_device
```

**Location**: [nvshmem4py/examples/init_fini.py:13-35](../nvshmem4py/examples/init_fini.py#L13-L35)

### Initialization Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│          mpi_uid_init(): Initialization Flow                   │
└────────────────────────────────────────────────────────────────┘

Rank 0                         Rank 1, 2, ...
  │                               │
  ├─> get_unique_id()             ├─> get_unique_id(empty=True)
  │    └─> Generate UUID          │    └─> Empty struct
  │        [0x1a2b3c4d...]         │        [0x00000000...]
  │                               │
  ├─────── MPI.Bcast(uid) ───────►│
  │                               │
  │    [0x1a2b3c4d...]  ──────────┼──► [0x1a2b3c4d...] (same!)
  │                               │
  ├─> nvshmem.core.init(uid, rank=0, nranks=4)
  │    │                          ├─> nvshmem.core.init(uid, rank=1, nranks=4)
  │    │                          │    │
  │    └─> Python layer           │    └─> Python layer
  │         └─> Cython bindings   │         └─> Cython bindings
  │              └─> C++ init     │              └─> C++ init
  │                   └─> Bootstrap (UID mode)
  │                        │      │                   │
  │                        ├─────┼─────┬─────────────┤
  │                        │     │     │             │
  │                     TCP Sockets for handshake:
  │                     Rank 0 ←→ Rank 1 ←→ Rank 2 ←→ Rank 3
  │                        │
  │                        └─> Exchange:
  │                             - Hostname
  │                             - Process ID
  │                             - GPU device ID
  │                             - Heap address
  │                             - Transport info
  │                        │
  │                        └─> All ranks synchronized
  │                             Ready for NVSHMEM operations!
  │                               │
  └─────────────────────────────┼─────────────────────────────────►
                                │
                          PE 0, PE 1, PE 2, PE 3 initialized
```

### Alternative: Torchrun UID Init

```python
# [8] ═══════════════════════════════════════════════════════════
# EXAMPLE 2: Torchrun UID-based Initialization
# ═══════════════════════════════════════════════════════════════
def torchrun_uid_init_bcast():
    """
    Initialize NVSHMEM using UniqueID with `torchrun` as the launcher
    Uses torch.distributed.broadcast to distribute UID
    """
    # [9] Set Torch device
    local_rank = int(os.environ['LOCAL_RANK'])  # LINE 44
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # nvshmem4py requires a cuda.core Device at init time
    dev = Device(local_rank)            # LINE 49
    dev.set_current()
    stream = dev.create_stream()

    # [10] ═══════════════════════════════════════════════════════════
    # INITIALIZE TORCH DISTRIBUTED (for UID broadcast)
    # ═══════════════════════════════════════════════════════════════
    world_size = torch.cuda.device_count()
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",   # LINE 57
        rank=local_rank,
        world_size=world_size,
        device_id=device
    )
    # CALL STACK:
    #   └─> torch.distributed.init_process_group() [PyTorch C++ backend]
    #        ├─> Initialize NCCL for GPU communication
    #        └─> Initialize Gloo for CPU fallback

    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    # [11] ═══════════════════════════════════════════════════════════
    # CREATE AND BROADCAST UID (using PyTorch)
    # ═══════════════════════════════════════════════════════════════
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank_id == 0:
        uniqueid = nvshmem.core.get_unique_id()  # LINE 71

    # Convert NumPy array to PyTorch tensor
    data = torch.tensor(uniqueid._data)  # LINE 74

    # Use torch.distributed.broadcast to send the UID to all ranks
    dist.broadcast(data, src=0)          # LINE 76
    # CALL STACK:
    #   └─> torch.distributed.broadcast() [PyTorch]
    #        └─> NCCL broadcast (if GPU tensor) or Gloo (if CPU)
    #             └─> ncclBroadcast() [NCCL library]

    dist.barrier()                       # LINE 77

    if rank_id != 0:
        uniqueid._data = data.numpy()    # LINE 80

    # [12] ═══════════════════════════════════════════════════════════
    # INITIALIZE NVSHMEM
    # ═══════════════════════════════════════════════════════════════
    nvshmem.core.init(
        device=dev,
        uid=uniqueid,
        rank=rank_id,
        nranks=num_ranks,
        initializer_method="uid"
    )  # LINE 82
    # Same call path as MPI example above
```

**Key Difference**: Uses PyTorch's distributed primitives instead of MPI for UID broadcast.

---

## Example 2: on-stream.py - Stream-Based Collectives

### Purpose

Demonstrates **asynchronous stream-based collectives** in Python:
- `nvshmem.core.reduce()` on CUDA stream
- Integration with Numba kernels
- Memory management with `nvshmem.core.array()`

### Source Code with Annotations

```python
# nvshmem4py/examples/on-stream.py

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# See License.txt for license information

"""
This file implements `examples/on-stream.cu` in Python
"""
import cupy
from numba import cuda
from cuda.core.experimental import Device, system

import nvshmem.core

from mpi4py import MPI

THRESHOLD = 42
CORRECTION = 7

# [1] ═══════════════════════════════════════════════════════════
# KERNEL 1: Local accumulation using Numba
# ═══════════════════════════════════════════════════════════════
@cuda.jit
def accumulate(input, partial_sum):
    """
    Accumulate kernel: Input is a 1-d array and partial_sum is a 1x1 array
    """
    index = cuda.threadIdx.x              # LINE 30
    if index == 0:
        partial_sum[0] = 0
    cuda.syncthreads()
    numba.cuda.atomic.add(partial_sum, 0, input[index])  # LINE 34
    # COMPILATION:
    #   └─> @cuda.jit decorator [Numba]
    #        ├─> Compile Python → LLVM IR
    #        ├─> Link with CUDA runtime
    #        └─> Generate PTX → SASS

# [2] ═══════════════════════════════════════════════════════════
# KERNEL 2: Conditional correction based on global sum
# ═══════════════════════════════════════════════════════════════
@cuda.jit
def correct_accumulate(input, partial_sum, full_sum):
    index = cuda.threadIdx.x
    if (full_sum > THRESHOLD):        # LINE 39
        input[index] = input[index] - CORRECTION
    if index == 0:
        partial_sum[0] = 0
    cuda.syncthreads()
    numba.cuda.atomic.add(partial_sum, 0, input[index])  # LINE 44

# [3] ═══════════════════════════════════════════════════════════
# MAIN: Initialize and run reduction
# ═══════════════════════════════════════════════════════════════
# Initialize NVSHMEM Using an MPI communicator
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices  # LINE 47
dev = Device(local_rank_per_node)
dev.set_current()
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD,
                   initializer_method="mpi")  # LINE 50
# CALL STACK:
#   └─> nvshmem.core.init() [nvshmem4py/nvshmem/core/init_fini.py:121+]
#        │
#        ├─> Validation: Check mpi_comm is mpi4py.MPI.Comm
#        │
#        ├─> Build init attributes:
#        │    └─> bindings.InitAttr()
#        │         ├─> Set bootstrap flags: NVSHMEMX_INIT_WITH_MPI_COMM
#        │         └─> bindings.set_attr_mpi_comm(mpi_comm, attr)
#        │              │
#        │              └─> [nvshmem4py/nvshmem/bindings/cynvshmem.pyx:850+]
#        │                   Extract MPI_Comm* from mpi4py object
#        │                   Store in attr->args.mpi_args.comm
#        │
#        └─> bindings.init_thread(NVSHMEM_THREAD_SERIALIZED, attr)
#             │
#             └─> nvshmemi_init_thread() [src/host/init/init.cu:500+]
#                  ├─> Bootstrap init (MPI mode):
#                  │    └─> [src/modules/bootstrap/mpi/bootstrap_mpi.cpp:100+]
#                  │         ├─> MPI_Comm_rank(comm, &rank)
#                  │         ├─> MPI_Comm_size(comm, &nranks)
#                  │         ├─> MPI_Allgather() to exchange PE info
#                  │         └─> Build connectivity map
#                  │
#                  └─> Continue with topology, heap, transports, etc.

mype = nvshmem.core.my_pe()            # LINE 52
npes = nvshmem.core.n_pes()            # LINE 53
mype_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)  # LINE 54
# CALL STACK for my_pe():
#   └─> nvshmem.core.my_pe() [nvshmem4py/nvshmem/core/direct.py:50+]
#        └─> bindings.my_pe()
#             │
#             └─> [nvshmem4py/nvshmem/bindings/cynvshmem.pyx:27-28]
#                  cdef int nvshmem_my_pe():
#                      return _nvshmem._nvshmem_my_pe()
#                  │
#                  └─> nvshmem_my_pe() [inline in src/include/host/nvshmem_api.h:75]
#                       └─> return nvshmemi_state->mype

# [4] ═══════════════════════════════════════════════════════════
# ALLOCATE SYMMETRIC MEMORY
# ═══════════════════════════════════════════════════════════════
input_nelems = 512
to_all_elems = 1
stream = dev.create_stream()          # LINE 58

input = nvshmem.core.array((input_nelems,), dtype="int")  # LINE 59
# CALL STACK:
#   └─> nvshmem.core.array() [nvshmem4py/nvshmem/core/memory.py:200+]
#        │
#        ├─> Calculate size: size = input_nelems * sizeof(int)
#        │
#        ├─> Allocate via buffer():
#        │    └─> nvshmem.core.buffer(size) [memory.py:54+]
#        │         │
#        │         ├─> Get or create NvshmemResource for device
#        │         │    └─> _mr_references[dev_id] = NvshmemResource(dev)
#        │         │
#        │         └─> resource.allocate(size)
#        │              │
#        │              └─> [nvshmem4py/nvshmem/core/nvshmem_types.py:100+]
#        │                   class NvshmemResource:
#        │                       def allocate(self, size):
#        │                           ptr = bindings.nvshmem_malloc(size)
#        │                           │
#        │                           └─> [cynvshmem.pyx:43-44]
#        │                                cdef void* nvshmem_malloc(size):
#        │                                    return _nvshmem._nvshmem_malloc(size)
#        │                                │
#        │                                └─> nvshmem_malloc(size)
#        │                                     [src/host/mem/mem.cpp:350+]
#        │                                     ├─> dlmalloc(size) from symmetric heap
#        │                                     ├─> Register with transports
#        │                                     └─> Barrier (collective allocation)
#        │
#        │                           # Wrap in cuda.core.Buffer
#        │                           return Buffer(ptr, size, resource=self)
#        │
#        └─> Wrap Buffer in NumPy/CuPy-compatible array view
#             └─> cuda.core.Buffer supports DLPack (__dlpack__)
#                  └─> Can convert to CuPy: cupy.from_dlpack(buffer)

partial_sum = nvshmem.core.array((1,), dtype="int")      # LINE 60
full_sum = nvshmem.core.array((1,), dtype="int")         # LINE 61

# [5] ═══════════════════════════════════════════════════════════
# LAUNCH KERNELS AND COLLECTIVE
# ═══════════════════════════════════════════════════════════════
accumulate[1, input_nelems, 0, stream](input, partial_sum)  # LINE 63
# CALL STACK:
#   └─> Numba kernel launch [Numba runtime]
#        ├─> Compile kernel (if not cached)
#        │    ├─> Python AST → LLVM IR
#        │    ├─> Optimize IR
#        │    └─> LLVM IR → PTX → SASS
#        │
#        └─> cudaLaunchKernel(accumulate_kernel, grid=(1,), block=(512,),
#                              args=[input.ptr, partial_sum.ptr], stream=stream)

# [6] ═══════════════════════════════════════════════════════════
# STREAM-BASED REDUCTION
# ═══════════════════════════════════════════════════════════════
nvshmem.core.reduce(
    nvshmem.core.Teams.TEAM_WORLD,    # LINE 64
    full_sum,
    partial_sum,
    "sum",
    stream=stream
)
# CALL STACK:
#   └─> nvshmem.core.reduce() [nvshmem4py/nvshmem/core/collective.py:200+]
#        │
#        ├─> Validation:
#        │    ├─> Check full_sum, partial_sum are Buffers or arrays
#        │    ├─> Check dtype compatibility
#        │    └─> Check op is valid ("sum", "max", "min")
#        │
#        ├─> Extract buffers and metadata:
#        │    ├─> full_sum_buf = full_sum._buffer (cuda.core.Buffer)
#        │    ├─> partial_sum_buf = partial_sum._buffer
#        │    ├─> dtype_str = "int"  # NVSHMEM dtype name
#        │    └─> nelem = 1
#        │
#        └─> collective_on_buffer("reduce", team, full_sum_buf, partial_sum_buf,
#                                  dtype="int", op="sum", stream=stream)
#             │
#             └─> [nvshmem4py/nvshmem/core/collective.py:112+]
#                  │
#                  ├─> Build function name:
#                  │    name = f"nvshmemx_{dtype}_{op}_reduce_on_stream"
#                  │         = "nvshmemx_int_sum_reduce_on_stream"
#                  │
#                  ├─> Get Cython binding:
#                  │    func = getattr(bindings, name)
#                  │    │
#                  │    └─> bindings.nvshmemx_int_sum_reduce_on_stream
#                  │         [nvshmem4py/nvshmem/bindings/cynvshmem.pyx:195-196]
#                  │         cdef int nvshmemx_int_sum_reduce_on_stream(...):
#                  │             return _nvshmem._nvshmemx_int_sum_reduce_on_stream(...)
#                  │         │
#                  │         └─> [_internal/nvshmem.pyx]
#                  │              cdef extern from "nvshmemx.h":
#                  │                  int _nvshmemx_int_sum_reduce_on_stream(
#                  │                      nvshmem_team_t team, int *dest,
#                  │                      const int *src, size_t nelem,
#                  │                      cudaStream_t stream
#                  │                  ) nogil
#                  │              │
#                  │              └─> C++ CALL:
#                  │                   nvshmemx_int_sum_reduce_on_stream(...)
#                  │                   [src/host/coll/rdxn/rdxn_on_stream.cpp:100+]
#                  │                   │
#                  │                   ├─> Choose algorithm:
#                  │                   │    ├─> NVLS (if Hopper+, same node)
#                  │                   │    ├─> Ring (if small team)
#                  │                   │    └─> Tree (if large distributed)
#                  │                   │
#                  │                   └─> Launch kernel:
#                  │                        reduce_kernel<int><<<grid, block, 0, stream>>>(
#                  │                            dest, src, nelem, NVSHMEM_SUM, ...
#                  │                        )
#                  │                        │
#                  │                        └─> [src/include/non_abi/device/coll/reducescatter.cuh:50+]
#                  │                             __global__ void reduce_kernel(...) {
#                  │                                 // Multi-round reduction
#                  │                                 for (int round = 0; round < log2(npes); round++) {
#                  │                                     int peer = compute_peer();
#                  │                                     nvshmem_int_put(scratch, local_sum, 1, peer);
#                  │                                     nvshmem_quiet();
#                  │                                     local_sum += scratch[0];
#                  │                                 }
#                  │                                 dest[0] = local_sum;
#                  │                             }
#                  │
#                  └─> Return status code to Python
#                       if status != 0:
#                           raise NvshmemError(...)

correct_accumulate[1, input_nelems, 0, stream](input, partial_sum, full_sum)  # LINE 66
stream.sync()                         # LINE 67

print(f"[{mype} of {npes}] Run complete")  # LINE 69

# [7] ═══════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════
nvshmem.core.free_array(input)       # LINE 71
nvshmem.core.free_array(partial_sum) # LINE 72
nvshmem.core.free_array(full_sum)    # LINE 73
# CALL STACK:
#   └─> nvshmem.core.free_array() [nvshmem4py/nvshmem/core/memory.py:300+]
#        └─> nvshmem.core.free(array._buffer)
#             │
#             └─> [memory.py:94+]
#                  ├─> Mark buffer as freed in resource
#                  └─> buffer.close()
#                       └─> bindings.nvshmem_free(buffer.ptr)
#                            │
#                            └─> nvshmem_free(ptr) [src/host/mem/mem.cpp:400+]
#                                 ├─> Unregister from transports
#                                 ├─> dlfree(ptr) in symmetric heap
#                                 └─> Barrier (collective free)

nvshmem.core.finalize()               # LINE 74
```

**Location**: [nvshmem4py/examples/on-stream.py](../nvshmem4py/examples/on-stream.py)

### Complete Call Stack: `nvshmem.core.reduce()`

```
Python User Code:
  nvshmem.core.reduce(team, full_sum, partial_sum, "sum", stream=stream)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Python Core API                                        │
│ [nvshmem4py/nvshmem/core/collective.py:200+]                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ├─> 1. Type validation
                        │    ├─> Check Buffers/arrays
                        │    ├─> Extract dtype
                        │    └─> Validate operation
                        │
                        ├─> 2. Extract buffer pointers
                        │    └─> full_sum._ptr, partial_sum._ptr
                        │
                        └─> 3. Call collective_on_buffer()
                             │
                             ├─> Build function name:
                             │    "nvshmemx_int_sum_reduce_on_stream"
                             │
                             └─> getattr(bindings, func_name)(...)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Cython Wrapper                                         │
│ [nvshmem4py/nvshmem/bindings/cynvshmem.pyx:195+]               │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        │ cdef int nvshmemx_int_sum_reduce_on_stream(
                        │     nvshmem_team_t team, int *dest, const int *src,
                        │     size_t nelem, cudaStream_t stream
                        │ ) except* nogil:
                        │     return _nvshmem._nvshmemx_int_sum_reduce_on_stream(
                        │         team, dest, src, nelem, stream
                        │     )
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Cython Internal (C++ Extern Declaration)               │
│ [nvshmem4py/nvshmem/bindings/_internal/nvshmem.pyx]            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        │ cdef extern from "nvshmemx.h":
                        │     int _nvshmemx_int_sum_reduce_on_stream(
                        │         nvshmem_team_t team, int *dest,
                        │         const int *src, size_t nelem,
                        │         cudaStream_t stream
                        │     ) nogil
                        │
                        │ (Direct C function call - no Python overhead)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: C++ Host API                                           │
│ [src/include/host/nvshmemx_coll_api.h:500+]                    │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        │ int nvshmemx_int_sum_reduce_on_stream(
                        │     nvshmem_team_t team, int *dest,
                        │     const int *source, size_t nreduce,
                        │     cudaStream_t stream
                        │ ) {
                        │     return nvshmemi_rdxn_on_stream<int>(
                        │         team, dest, source, nreduce,
                        │         NVSHMEM_SUM, stream
                        │     );
                        │ }
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: C++ Implementation                                     │
│ [src/host/coll/rdxn/rdxn_on_stream.cpp:100+]                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ├─> Algorithm selection:
                        │    if (use_nvls()) → NVLS path
                        │    else if (small_team()) → Ring
                        │    else → Tree
                        │
                        └─> Launch CUDA kernel:
                             reduce_kernel<int><<<grid, block, 0, stream>>>(
                                 dest, source, nreduce, NVSHMEM_SUM, team_info
                             )
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 6: Device Kernel                                          │
│ [src/include/non_abi/device/coll/reducescatter.cuh:50+]        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        │ __global__ void reduce_kernel(...) {
                        │     int mype = nvshmem_my_pe();
                        │     int npes = nvshmem_n_pes();
                        │     int local_sum = source[tid];
                        │
                        │     // Multi-round reduction
                        │     for (int round = 0; round < log2(npes); round++) {
                        │         int peer = (mype ^ (1 << round));
                        │         nvshmem_int_put(scratch, &local_sum, 1, peer);
                        │         nvshmem_quiet();
                        │         local_sum += scratch[0];  // Reduce
                        │     }
                        │
                        │     if (tid == 0) dest[0] = local_sum;
                        │ }
                        │
                        └─> Uses nvshmem_int_put() (device RMA)
                             │
                             └─> See CPP_EXAMPLES_WALKTHROUGH.md
                                  for device RMA call stack
```

### Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│         on-stream.py: Reduction Execution Flow                 │
└────────────────────────────────────────────────────────────────┘

Python Host (CPU)                         CUDA Stream (GPU)
─────────────────────────────────────────────────────────────────
  │
  ├─> accumulate[1, 512, 0, stream](...)
  │    └─> Numba compiles & launches
  │                                        ┌────────────────────┐
  │                                        │ accumulate kernel  │
  │                                        │ Compute local sum  │
  │                                        │ partial_sum[0] = X │
  │                                        └────────────────────┘
  │                                                 │
  ├─> nvshmem.core.reduce(team, full_sum, partial_sum, "sum", stream)
  │    │                                            │
  │    ├─> Python validation                       │
  │    ├─> Cython call                             │
  │    └─> C++ launches kernel ─────────────────► ┌────────────────────┐
  │                                                │ reduce_kernel      │
  │                                                │ Round 0:           │
  │                                                │   PUT to PE (^ 1)  │──┐
  │                                                │   Wait & reduce    │  │
  │                                                │ Round 1:           │  │
  │                                                │   PUT to PE (^ 2)  │──┼─► NVLink/IB
  │                                                │   Wait & reduce    │  │
  │                                                │ Result: full_sum   │◄─┘
  │                                                └────────────────────┘
  │                                                         │
  ├─> correct_accumulate[1, 512, 0, stream](...)           │
  │    └─> Numba launches                                  │
  │                                        ┌────────────────▼───┐
  │                                        │ correct_accumulate │
  │                                        │ Read full_sum      │
  │                                        │ Apply correction   │
  │                                        └────────────────────┘
  │                                                 │
  └─> stream.sync() ◄─────────────────────────────┘
       Host blocks until all GPU work done
```

---

## Example 3: simple_p2p_kernel.py - Basic RMA

### Purpose

Demonstrates **simple point-to-point RMA** in Python with Numba device code.

### Source Code with Annotations

```python
# nvshmem4py/examples/simple_p2p_kernel.py

import cupy
import nvshmem.core
from cuda.core.experimental import Device, system
from numba import cuda

# [1] ═══════════════════════════════════════════════════════════
# DEVICE KERNEL: Simple shift using Numba
# ═══════════════════════════════════════════════════════════════
@cuda.jit
def simple_shift(arr, dst_pe):
    arr[0] = dst_pe  # Write destination PE ID
    # NOTE: This is a simple example - real device RMA shown in Example 4

# [2] Initialize NVSHMEM Using an MPI communicator
from mpi4py import MPI
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()
stream = dev.create_stream()
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD,
                   initializer_method="mpi")

# [3] ═══════════════════════════════════════════════════════════
# ALLOCATE SYMMETRIC ARRAY (Helper function)
# ═══════════════════════════════════════════════════════════════
array = nvshmem.core.array((1,), dtype="int32")
# Returns a cuda.core.Buffer wrapped as array
# Backed by nvshmem_malloc() symmetric memory

my_pe = nvshmem.core.my_pe()

# [4] Compute destination PE (unidirectional ring)
dst_pe = (my_pe + 1) % nvshmem.core.n_pes()

# [5] ═══════════════════════════════════════════════════════════
# GET PEER ARRAY (for direct load/store access)
# ═══════════════════════════════════════════════════════════════
# This function returns an Array which can be directly load/store'd to over NVLink
# The dst_PE must be in the same NVL domain as the PE calling this function
dev_dst = nvshmem.core.get_peer_array(array, dst_pe)
# CALL STACK:
#   └─> nvshmem.core.get_peer_array() [nvshmem4py/nvshmem/core/memory.py:125+]
#        │
#        ├─> Validate:
#        │    ├─> Check array is nvshmem-allocated
#        │    └─> Check dst_pe is valid
#        │
#        ├─> Get remote address:
#        │    └─> remote_ptr = bindings.nvshmem_ptr(array._ptr, dst_pe)
#        │         │
#        │         └─> nvshmem_ptr() [src/include/host/nvshmem_api.h:88]
#        │              __device__ __host__ void *nvshmem_ptr(const void *dest, int pe) {
#        │                  if (peer_heap_base_accessible[pe]) {
#        │                      # Calculate remote address
#        │                      return peer_heap_base[pe] +
#        │                             (dest - heap_base);
#        │                  } else {
#        │                      return NULL;  # Not P2P accessible
#        │                  }
#        │              }
#        │
#        └─> Wrap remote pointer in Buffer:
#             └─> Buffer(remote_ptr, array.size, memory_resource=PeerResource(pe))
#                  # PeerResource marks this as peer memory (no explicit free)

# [6] ═══════════════════════════════════════════════════════════
# LAUNCH KERNEL
# ═══════════════════════════════════════════════════════════════
block = 1
grid = (array.size + block - 1) // block
simple_shift[block, grid, 0, 0](array, my_pe)

# [7] ═══════════════════════════════════════════════════════════
# BARRIER (synchronize across all PEs)
# ═══════════════════════════════════════════════════════════════
nvshmem.core.barrier(nvshmem.core.Teams.TEAM_NODE, stream)
# CALL STACK:
#   └─> nvshmem.core.barrier() [nvshmem4py/nvshmem/core/collective.py:350+]
#        └─> bindings.barrier_on_stream(team._handle, stream._handle)
#             │
#             └─> nvshmemx_barrier_on_stream(team, stream)
#                  [src/host/coll/barrier/barrier_on_stream.cpp:50+]
#                  └─> Launch barrier kernel on stream
#                       └─> [src/include/non_abi/device/coll/barrier.cuh:50+]

# This should print the neighbor's PE ID
print(f"From PE {my_pe}, array contains {array}")

# [8] Cleanup
nvshmem.core.free_array(arr_src)
nvshmem.core.free_array(arr_dst)
nvshmem.core.finalize()
```

**Location**: [nvshmem4py/examples/simple_p2p_kernel.py](../nvshmem4py/examples/simple_p2p_kernel.py)

---

## Example 4: device_custom_kernel.py - Numba Device Code

### Purpose

Demonstrates **device-side NVSHMEM calls from Numba**:
- Import device functions from `nvshmem.bindings.device.numba`
- Call NVSHMEM RMA/sync from GPU kernel (JIT-compiled Python)
- Ring reduction implemented in Numba

### Source Code with Annotations

```python
# nvshmem4py/examples/device_custom_kernel.py

from numba import cuda
import nvshmem.core

# [1] ═══════════════════════════════════════════════════════════
# IMPORT DEVICE-SIDE NVSHMEM FUNCTIONS (for Numba)
# ═══════════════════════════════════════════════════════════════
from nvshmem.bindings.device.numba import (
    my_pe,                    # Device: nvshmem_my_pe()
    n_pes,                    # Device: nvshmem_n_pes()
    put_signal_nbi,           # Device: nvshmem_putmem_signal_nbi()
    signal_wait_until         # Device: nvshmem_signal_wait_until()
)
# IMPORTANT: These are DEVICE functions, callable only from @cuda.jit kernels

from nvshmem.core import SignalOp, ComparisonType
import cuda.core
from cuda.core.experimental import Device, system
from mpi4py import MPI

signal_op = SignalOp.ADD
comparison_type = ComparisonType.GE

SIGNAL_ADD = signal_op.value     # Convert enum to int
CMP_GE = comparison_type.value

# [2] ═══════════════════════════════════════════════════════════
# NUMBA DEVICE KERNEL: Ring Reduction
# Uses NVSHMEM device functions
# ═══════════════════════════════════════════════════════════════
@cuda.jit(lto=True)  # LTO=Link-Time Optimization (required for NVSHMEM linking)
def ring_reduce(dst, src, nreduce, signal, chunk_size):
    # [3] Query PE info from device (constant memory access)
    mype = my_pe()                # LINE 17
    # CALL STACK (device side):
    #   └─> my_pe() [nvshmem4py/nvshmem/bindings/device/numba/_numbast.py:50+]
    #        │
    #        └─> Numba intrinsic that generates:
    #             declare i32 @nvshmem_my_pe() #0
    #             │
    #             └─> Links to NVSHMEM device library (.bc file)
    #                  [src/device/init/init_device.cu:50+]
    #                  __device__ int nvshmem_my_pe() {
    #                      return nvshmemi_device_state_d.mype;
    #                  }
    #
    # nvshmemi_device_state_d is __constant__ memory
    # Populated during nvshmem_init()

    npes = n_pes()                # LINE 18
    peer = (mype + 1) % npes      # LINE 19

    thread_id = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    num_blocks = cuda.gridDim.x
    block_idx = cuda.blockIdx.x
    elems_per_block = nreduce // num_blocks

    # Adjust pointers for this block
    if elems_per_block * (block_idx + 1) > nreduce:
        return

    signal_block = signal[block_idx:block_idx+1]

    chunk_elems = chunk_size
    num_chunks = elems_per_block // chunk_elems

    # [4] ═══════════════════════════════════════════════════════════
    # REDUCE PHASE: Ring reduction with PUT+SIGNAL
    # ═══════════════════════════════════════════════════════════════
    block_base_offset = block_idx * elems_per_block
    for chunk, offset in enumerate(range(block_base_offset,
                                          block_base_offset + elems_per_block,
                                          chunk_elems)):
        src_block = src[offset:offset+chunk_elems]
        dst_block = dst[offset:offset+chunk_elems]

        if mype != 0:
            # [5] Wait for data from previous PE
            if thread_id == 0:
                signal_wait_until(signal_block, CMP_GE, chunk + 1)  # LINE 46
                # CALL STACK (device side):
                #   └─> signal_wait_until() [nvshmem4py/nvshmem/bindings/device/numba/_numbast.py:100+]
                #        │
                #        └─> Numba intrinsic → LLVM IR:
                #             declare void @nvshmem_signal_wait_until(
                #                 i64* %ptr, i32 %cmp, i64 %value
                #             )
                #             │
                #             └─> Links to NVSHMEM device library:
                #                  [src/include/non_abi/device/wait/nvshmemi_wait_until_apis.cuh:50+]
                #                  __device__ void nvshmem_uint64_wait_until(
                #                      uint64_t *ptr, int cmp, uint64_t value
                #                  ) {
                #                      while (1) {
                #                          uint64_t current = *ptr;
                #                          if (compare(current, cmp, value)) break;
                #                          __nanosleep(100);  # Yield
                #                      }
                #                  }

            cuda.syncthreads()

            # Reduce with received data
            for i in range(thread_id, chunk_elems, num_threads):
                dst_block[i] = dst_block[i] + src_block[i]
            cuda.syncthreads()

        # [6] Send partial result to next PE (with signal)
        if thread_id == 0:
            src_data = src_block if mype == 0 else dst_block
            put_signal_nbi(
                dst_block,        # Remote destination
                src_data,         # Local source
                chunk_elems,      # Number of elements
                signal_block,     # Signal address
                1,                # Signal value to add
                SIGNAL_ADD,       # Signal operation (ADD)
                peer              # Target PE
            )  # LINE 55-56
            # CALL STACK (device side):
            #   └─> put_signal_nbi() [nvshmem4py/nvshmem/bindings/device/numba/_numbast.py:200+]
            #        │
            #        └─> Numba intrinsic → LLVM IR:
            #             declare void @nvshmem_putmem_signal_nbi(
            #                 i8* %dest, i8* %src, i64 %size,
            #                 i64* %sig_addr, i64 %signal,
            #                 i32 %sig_op, i32 %pe
            #             )
            #             │
            #             └─> Links to NVSHMEM device library:
            #                  [src/include/non_abi/device/pt-to-pt/transfer_device.cuh:500+]
            #                  __device__ void nvshmem_putmem_signal_nbi(...) {
            #                      # Step 1: Perform PUT
            #                      nvshmemi_put<uint8_t>(dest, src, size, pe);
            #
            #                      # Step 2: Fence (ensure PUT visible)
            #                      nvshmem_fence();
            #
            #                      # Step 3: Signal remote PE
            #                      if (sig_op == NVSHMEM_SIGNAL_SET) {
            #                          nvshmem_uint64_p(sig_addr, signal, pe);
            #                      } else if (sig_op == NVSHMEM_SIGNAL_ADD) {
            #                          nvshmem_uint64_atomic_add(sig_addr, signal, pe);
            #                      }
            #                  }

    # [7] BROADCAST PHASE (omitted for brevity - similar pattern)
    # ...

# [8] ═══════════════════════════════════════════════════════════
# MAIN: Setup and launch
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    size = args.size
    chunk_size = args.chunk_size

    # [9] Initialize
    rank = MPI.COMM_WORLD.Get_rank()
    dev = cuda.Device(rank % system.num_devices)
    dev.set_current()
    stream = dev.create_stream()
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD,
                       initializer_method="mpi")

    # [10] Allocate symmetric arrays
    src = nvshmem.core.array((size,), dtype="float32")
    dst = nvshmem.core.array((size,), dtype="float32")
    src[:] = 1
    dst[:] = 0
    signal = nvshmem.core.array((1,), dtype="uint64")

    # [11] ═══════════════════════════════════════════════════════════
    # LAUNCH NUMBA KERNEL with NVSHMEM device calls
    # ═══════════════════════════════════════════════════════════════
    ring_reduce[1, 1](dst, src, size, signal, chunk_size)  # LINE 96
    # COMPILATION:
    #   └─> Numba JIT compilation [Numba runtime]
    #        │
    #        ├─> Step 1: Python AST → Numba IR
    #        │    └─> Parse @cuda.jit function
    #        │
    #        ├─> Step 2: Numba IR → LLVM IR
    #        │    ├─> Generate LLVM IR for kernel logic
    #        │    └─> Generate LLVM IR for NVSHMEM intrinsics:
    #        │         declare i32 @nvshmem_my_pe()
    #        │         declare void @nvshmem_putmem_signal_nbi(...)
    #        │         declare void @nvshmem_signal_wait_until(...)
    #        │
    #        ├─> Step 3: Link with NVSHMEM device library
    #        │    └─> Load NVSHMEM bitcode library (.bc file)
    #        │         Location: /path/to/libnvshmem_device.bc
    #        │         │
    #        │         └─> Numba's LLVM linker merges:
    #        │              ├─> ring_reduce LLVM IR
    #        │              └─> NVSHMEM device functions LLVM IR
    #        │                   (my_pe, put_signal_nbi, etc.)
    #        │
    #        ├─> Step 4: LLVM optimization passes
    #        │    ├─> Inline NVSHMEM functions (many are __forceinline__)
    #        │    ├─> Constant propagation
    #        │    └─> Loop optimization
    #        │
    #        ├─> Step 5: LLVM IR → PTX
    #        │    └─> NVPTX backend generates PTX code
    #        │
    #        └─> Step 6: PTX → SASS
    #             └─> CUDA driver JIT compiles PTX → native GPU code
    #
    # LAUNCH:
    #   └─> cudaLaunchKernel(ring_reduce, grid=(1,), block=(1,),
    #                         args=[dst._ptr, src._ptr, size, signal._ptr, chunk_size])

    print(dst)  # LINE 98
```

**Location**: [nvshmem4py/examples/device_custom_kernel.py](../nvshmem4py/examples/device_custom_kernel.py)

### Numba Compilation Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│    Numba Kernel Compilation with NVSHMEM Device Functions      │
└────────────────────────────────────────────────────────────────┘

Step 1: Python Source Code
────────────────────────────────────────────────────────────────
@cuda.jit(lto=True)
def ring_reduce(dst, src, nreduce, signal, chunk_size):
    mype = my_pe()  # NVSHMEM device function
    ...
    put_signal_nbi(dst, src, nelems, signal, 1, SIGNAL_ADD, peer)

Step 2: Numba IR (Intermediate Representation)
────────────────────────────────────────────────────────────────
ring_reduce:
  $mype = call @my_pe()
  ...
  call @put_signal_nbi($dst, $src, $nelems, ...)

Step 3: LLVM IR Generation
────────────────────────────────────────────────────────────────
define void @ring_reduce(float* %dst, float* %src, ...) {
entry:
  %mype = call i32 @nvshmem_my_pe()  ← NVSHMEM intrinsic
  ...
  call void @nvshmem_putmem_signal_nbi(
      i8* %dst, i8* %src, i64 %size,
      i64* %signal, i64 1, i32 1, i32 %peer
  )  ← NVSHMEM intrinsic
  ret void
}

# NVSHMEM intrinsics declared but not defined (external)
declare i32 @nvshmem_my_pe()
declare void @nvshmem_putmem_signal_nbi(i8*, i8*, i64, i64*, i64, i32, i32)

Step 4: Link with NVSHMEM Bitcode Library
────────────────────────────────────────────────────────────────
Numba's LLVM Linker:
  ├─> Load: ring_reduce LLVM IR (from Numba)
  └─> Load: libnvshmem_device.bc (NVSHMEM device library)
       │
       └─> Contains definitions:
            define i32 @nvshmem_my_pe() {
                %state = load @nvshmemi_device_state_d
                %mype = extractvalue %state, 0  # .mype field
                ret i32 %mype
            }

            define void @nvshmem_putmem_signal_nbi(...) {
                # Inline implementation of PUT + SIGNAL
                ...
            }

  ├─> Link: Merge both IR modules
  └─> Result: Single LLVM IR with all definitions resolved

Step 5: LLVM Optimization
────────────────────────────────────────────────────────────────
LLVM Optimization Passes:
  ├─> Inline small functions (many NVSHMEM functions are __forceinline__)
  ├─> Constant propagation
  ├─> Dead code elimination
  └─> Loop optimizations

Result: Optimized LLVM IR

Step 6: PTX Generation
────────────────────────────────────────────────────────────────
LLVM NVPTX Backend:
  LLVM IR → PTX (parallel thread execution assembly)

Example PTX snippet:
.entry ring_reduce(.param .u64 dst, .param .u64 src, ...) {
    ...
    # Load mype from constant memory
    ld.const.u32 %r1, [nvshmemi_device_state_d];
    ...
    # Call inline PUT
    st.global.f32 [%remote_addr], %value;
    ...
}

Step 7: SASS Generation (JIT by CUDA driver)
────────────────────────────────────────────────────────────────
CUDA Driver:
  PTX → SASS (native GPU machine code for specific architecture)

Example SASS:
LDC.64 R0, c[0x0][0x180];  // Load from constant memory
...
STG.E [R2], R4;             // Global store (PUT)
...

Step 8: Kernel Launch
────────────────────────────────────────────────────────────────
cudaLaunchKernel(ring_reduce_compiled, grid, block, args);
GPU executes native SASS code
```

---

## Numba Integration Deep Dive

### How NVSHMEM Device Functions Work in Numba

```
┌────────────────────────────────────────────────────────────────┐
│     NVSHMEM Device Functions in Numba: Implementation          │
└────────────────────────────────────────────────────────────────┘

Component 1: Numba Intrinsics Definition
────────────────────────────────────────────────────────────────
File: nvshmem4py/nvshmem/bindings/device/numba/_numbast.py

from numba import cuda
from numba.core import types
from numba.cuda import cudadecl, cudaimpl

# [1] Declare intrinsic signature
@cudadecl.registry.register_global(my_pe)
class MyPeDecl(cudadecl.ConcreteTemplate):
    key = my_pe
    cases = [types.int32()]  # Returns int32

# [2] Implement intrinsic (generate LLVM IR)
@cudaimpl.registry.lower(my_pe, types.int32)
def my_pe_impl(context, builder, sig, args):
    # Generate LLVM IR call
    fn = context.declare_external_function(
        builder.module,
        name="nvshmem_my_pe",
        signature=types.int32(),
        linkage="external"  # External linkage (defined in .bc file)
    )
    return builder.call(fn, [])

# Similar for all NVSHMEM device functions:
# - n_pes()
# - put_signal_nbi()
# - signal_wait_until()
# - etc.

Component 2: NVSHMEM Bitcode Library
────────────────────────────────────────────────────────────────
File: build/lib/libnvshmem_device.bc

LLVM Bitcode (binary LLVM IR) containing compiled device functions:

define i32 @nvshmem_my_pe() {
    %state = load @nvshmemi_device_state_d, align 8
    %mype = extractvalue %state, 0  ; Extract .mype field
    ret i32 %mype
}

define i32 @nvshmem_n_pes() {
    %state = load @nvshmemi_device_state_d, align 8
    %npes = extractvalue %state, 1  ; Extract .npes field
    ret i32 %npes
}

# More complex functions (PUT, GET, AMO, etc.) also included

Component 3: Numba Linking Configuration
────────────────────────────────────────────────────────────────
File: nvshmem4py/nvshmem/core/init_fini.py

def module_init(device_lib_path):
    """
    Register NVSHMEM device library with Numba's linker
    """
    from numba import cuda
    from numba.cuda.cudadrv import driver

    # Load NVSHMEM bitcode
    with open(device_lib_path, 'rb') as f:
        bitcode = f.read()

    # Register with Numba's CUDA driver
    cuda.cudadrv.linker.register_device_library(
        "nvshmem_device",
        bitcode,
        kind="ltoir"  # LLVM IR
    )

Called during nvshmem.core.init():
  └─> Ensures Numba can find NVSHMEM device functions when linking

Component 4: User Kernel Compilation
────────────────────────────────────────────────────────────────
User writes:

@cuda.jit(lto=True)  # LTO = Link-Time Optimization (required!)
def my_kernel(...):
    mype = my_pe()  # Calls NVSHMEM device function

Numba compilation:
  ├─> Generate LLVM IR for my_kernel
  │    └─> Includes: call i32 @nvshmem_my_pe()
  │
  ├─> Link with registered device libraries:
  │    └─> Numba's linker merges:
  │         ├─> my_kernel LLVM IR
  │         └─> nvshmem_device.bc (contains nvshmem_my_pe definition)
  │
  ├─> Optimize merged LLVM IR
  │    └─> Inline nvshmem_my_pe (it's __forceinline__)
  │
  └─> Generate PTX → SASS
```

### Why `lto=True` is Required

```
Without lto=True:
────────────────────────────────────────────────────────────────
@cuda.jit  # No LTO
def my_kernel():
    mype = my_pe()  # External function call

Problem:
  - Numba generates PTX with undefined symbol: nvshmem_my_pe
  - PTX → SASS compilation fails (symbol not found)
  - Error: "undefined reference to nvshmem_my_pe"

With lto=True:
────────────────────────────────────────────────────────────────
@cuda.jit(lto=True)  # LTO enabled
def my_kernel():
    mype = my_pe()  # Will be linked

Solution:
  - Numba generates LLVM IR (not PTX yet)
  - Links LLVM IR with nvshmem_device.bc
  - Resolves all symbols
  - Then generates PTX → SASS
  - Success!
```

---

## Summary: Key Patterns in Python Examples

### 1. Initialization Pattern

```python
from cuda.core.experimental import Device
import nvshmem.core

dev = Device(local_rank)
dev.set_current()
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD,
                   initializer_method="mpi")
```

### 2. Memory Allocation Pattern

```python
# Symmetric array allocation
arr = nvshmem.core.array((size,), dtype="float32")
# Backed by nvshmem_malloc()
# Returns cuda.core.Buffer (DLPack-compatible)

# Cleanup
nvshmem.core.free_array(arr)
```

### 3. Stream-Based Collective Pattern

```python
stream = dev.create_stream()

# Launch kernel
kernel[grid, block, 0, stream](...)

# Collective on stream
nvshmem.core.reduce(team, dest, src, "sum", stream=stream)

# Synchronize
stream.sync()
```

### 4. Numba Device Code Pattern

```python
from nvshmem.bindings.device.numba import my_pe, put_signal_nbi

@cuda.jit(lto=True)  # LTO required for NVSHMEM linking!
def my_kernel(data, peer):
    mype = my_pe()  # Device-side NVSHMEM call
    put_signal_nbi(data, data, size, signal, 1, ADD, peer)
```

### 5. Finalization Pattern

```python
nvshmem.core.finalize()
# Automatically frees leaked buffers
# Calls nvshmemi_finalize() in C++
```

---

## Cross-Reference

For C++ examples and architecture details, see:

- **C++ Examples**: [CPP_EXAMPLES_WALKTHROUGH.md](./CPP_EXAMPLES_WALKTHROUGH.md)
- **System Architecture**: [ARCHITECTURE.md](./ARCHITECTURE.md)

For detailed component explanations:

- **Memory Management**: [ARCHITECTURE.md § Memory Management](./ARCHITECTURE.md#2-memory-management)
- **Collective Operations**: [ARCHITECTURE.md § Collective Operations](./ARCHITECTURE.md#4-collective-operations)
- **Transport Layer**: [ARCHITECTURE.md § Transport Layer](./ARCHITECTURE.md#6-transport-layer)
- **Python Bridge**: [ARCHITECTURE.md § Python-Specific Architecture](./ARCHITECTURE.md#python-specific-architecture)
