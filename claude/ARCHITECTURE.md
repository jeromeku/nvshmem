# NVSHMEM Architecture

> **Purpose**: This document describes the internal architecture of NVSHMEM (NVIDIA Shared Memory), a parallel programming interface based on OpenSHMEM that provides efficient multi-GPU communication primitives.
>
> **Audience**: Developers who want to understand NVSHMEM's internals, from user-facing APIs through abstraction layers to actual CUDA kernel calls.
>
> **Version**: NVSHMEM 3.5.0

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [System Architecture](#system-architecture)
3. [Data Flow: User API → Device Execution](#data-flow-user-api--device-execution)
4. [Core Subsystems](#core-subsystems)
5. [Multi-Device Synchronization](#multi-device-synchronization)
6. [Code Organization](#code-organization)
7. [Key Data Structures](#key-data-structures)
8. [Important Design Patterns](#important-design-patterns)

---

## High-Level Overview

### What is NVSHMEM?

NVSHMEM is a **Partitioned Global Address Space (PGAS)** library for multi-GPU systems that enables:
- **Direct GPU-to-GPU communication** without CPU involvement
- **Symmetric memory allocation** across Processing Elements (PEs)
- **One-sided RMA (Remote Memory Access)** operations
- **Atomic operations** across GPUs
- **Collective operations** (broadcast, reduce, barrier, etc.)
- **Multi-transport support** (CUDA P2P, InfiniBand, UCX, libfabric)

### Key Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                    NVSHMEM Conceptual Model                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PE 0           PE 1           PE 2           PE 3          │
│  ┌────┐         ┌────┐         ┌────┐         ┌────┐       │
│  │GPU0│         │GPU1│         │GPU2│         │GPU3│       │
│  └────┘         └────┘         └────┘         └────┘       │
│    │              │              │              │           │
│    ├─Symmetric────┼─Symmetric────┼─Symmetric────┼───────┐  │
│    │  Heap        │  Heap        │  Heap        │  Heap  │  │
│    │  [Addr X]    │  [Addr X]    │  [Addr X]    │  [Addr X] │
│    │              │              │              │           │
│    └──────────────┴──────────────┴──────────────┘           │
│           ^                                                  │
│           │                                                  │
│    nvshmem_put(dest, src, size, pe=2)                       │
│           └──────────────────────► Direct GPU access        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Processing Element (PE)**: A unit of execution, typically one GPU
**Symmetric Heap**: Memory allocated at the same virtual address on all PEs
**One-sided Communication**: PE can read/write remote memory without remote PE involvement

---

## System Architecture

### Architectural Layers

NVSHMEM is organized in a **layered architecture**:

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 0: USER APPLICATIONS                                       │
│ ┌────────────────────────────┬──────────────────────────────┐   │
│ │  C++ Applications          │  Python Applications         │   │
│ │  (examples/*.cu)           │  (nvshmem4py/examples/*.py)  │   │
│ └────────────┬───────────────┴──────────────┬───────────────┘   │
└──────────────┼──────────────────────────────┼───────────────────┘
               │                               │
┌──────────────┼──────────────────────────────┼───────────────────┐
│ Layer 1: PUBLIC API                          │                   │
│ ┌────────────▼───────────────┐  ┌───────────▼────────────────┐ │
│ │ C++ Host API               │  │ Python Core API            │ │
│ │ nvshmem.h, nvshmemx.h      │  │ nvshmem.core.*             │ │
│ │ - nvshmem_init()           │  │ - init()                   │ │
│ │ - nvshmem_malloc()         │  │ - buffer()                 │ │
│ │ - nvshmem_put()            │  │ - put()                    │ │
│ └────────────┬───────────────┘  └───────────┬────────────────┘ │
│              │                               │                   │
│              │                   ┌───────────▼────────────────┐ │
│              │                   │ Cython Bindings            │ │
│              │                   │ cynvshmem.pyx              │ │
│              │                   └───────────┬────────────────┘ │
└──────────────┼──────────────────────────────┼───────────────────┘
               │                               │
┌──────────────┼──────────────────────────────┼───────────────────┐
│ Layer 2: HOST IMPLEMENTATION (CPU-side)      │                   │
│ ┌────────────▼───────────────────────────────▼────────────────┐ │
│ │ Host Implementation (src/host/)                             │ │
│ │ ┌──────────────┐ ┌───────────┐ ┌────────────┐             │ │
│ │ │ init/        │ │ mem/      │ │ comm/      │             │ │
│ │ │ init.cu      │ │ mem.cpp   │ │ putget.cpp │             │ │
│ │ │ bootstrap    │ │ heap mgmt │ │ RMA ops    │             │ │
│ │ └──────────────┘ └───────────┘ └────────────┘             │ │
│ │ ┌──────────────┐ ┌───────────┐ ┌────────────┐             │ │
│ │ │ coll/        │ │ proxy/    │ │ team/      │             │ │
│ │ │ collectives  │ │ proxy.cpp │ │ teams      │             │ │
│ │ └──────────────┘ └───────────┘ └────────────┘             │ │
│ └───────────────────────────┬─────────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│ Layer 3: TRANSPORT ABSTRACTION                                  │
│ ┌───────────────────────────▼─────────────────────────────────┐ │
│ │ Transport Layer (src/host/transport/ + src/modules/transport)│ │
│ │ ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐           │ │
│ │ │ P2P      │ │ IBGDA  │ │ IBDEVX │ │ UCX      │  ...      │ │
│ │ │ NVLink   │ │ (IB+   │ │ (IB    │ │ Unified  │           │ │
│ │ │ PCIe     │ │  GDR)  │ │ DevX)  │ │ Comm X)  │           │ │
│ │ └──────────┘ └────────┘ └────────┘ └──────────┘           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│ Layer 4: DEVICE IMPLEMENTATION (GPU-side)                       │
│ ┌───────────────────────────▼─────────────────────────────────┐ │
│ │ Device-Side API (src/include/device/ + non_abi/device/)    │ │
│ │ ┌──────────────────┐ ┌─────────────────┐                   │ │
│ │ │ nvshmem_defines.h│ │ transfer_device │                   │ │
│ │ │ - nvshmem_put()  │ │ - nvshmemi_put<>│                   │ │
│ │ │ - nvshmem_get()  │ │ - nvshmemi_get<>│                   │ │
│ │ │ (inline)         │ │ (templates)     │                   │ │
│ │ └──────────────────┘ └─────────────────┘                   │ │
│ │ ┌──────────────────┐ ┌─────────────────┐                   │ │
│ │ │ Device AMO       │ │ Device Colls    │                   │ │
│ │ │ Atomics          │ │ Barriers, etc.  │                   │ │
│ │ └──────────────────┘ └─────────────────┘                   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ Layer 5: HARDWARE                                               │
│ ┌─────────────┐ ┌──────────────┐ ┌─────────────────────────┐  │
│ │ CUDA Cores  │ │ NVLink/PCIe  │ │ InfiniBand NICs (GPUDirect) │
│ └─────────────┘ └──────────────┘ └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Interfaces

| Interface | Location | Purpose |
|-----------|----------|---------|
| **C++ Host API** | `src/include/host/nvshmem_api.h` | User-facing host functions |
| **Device API** | `src/include/device/nvshmem_defines.h` | User-facing device (kernel) functions |
| **Python Core** | `nvshmem4py/nvshmem/core/` | High-level Python interface |
| **Cython Bindings** | `nvshmem4py/nvshmem/bindings/cynvshmem.pyx` | C++ ↔ Python bridge |
| **Transport API** | `src/include/internal/host_transport/transport.h` | Pluggable transport backends |
| **Bootstrap API** | `src/include/internal/bootstrap_host/nvshmemi_bootstrap.h` | PE discovery & initialization |

---

## Data Flow: User API → Device Execution

This section traces the **complete call path** from user code to actual GPU execution, for both C++ and Python.

### C++ API Call Path: `nvshmem_put()`

Let's trace a typical RMA (Remote Memory Access) operation:

#### 1. **User Code** (Application Layer)

```cpp
// examples/dev-guide-ring.cu
float *destination = (float *)nvshmem_malloc(sizeof(float) * num_elems);
nvshmem_float_put(destination, source, num_elems, peer_pe);
```

**Location**: `examples/dev-guide-ring.cu:23`

#### 2. **Device-Side API** (Inline/Template)

```cpp
// src/include/device/nvshmem_defines.h:98-103
#define NVSHMEMI_TYPENAME_PUT_IMPL(TYPENAME, TYPE)
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmem_##TYPENAME##_put(
        TYPE *dest, const TYPE *source, size_t nelems, int pe) {
        nvshmemi_put<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems, pe);
    }
```

**Expansion for `float`**:
```cpp
__device__ __forceinline__ void nvshmem_float_put(float *dest, const float *source,
                                                    size_t nelems, int pe) {
    nvshmemi_put<float, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems, pe);
}
```

**Location**: `src/include/device/nvshmem_defines.h:98-103`

#### 3. **Device Transfer Template** (Core Logic)

```cpp
// src/include/non_abi/device/pt-to-pt/transfer_device.cuh
template <typename T, NVSHMEMI_THREADGROUP_TYPE SCOPE>
__device__ __forceinline__ void nvshmemi_put(T *dest, const T *source,
                                              size_t nelems, int pe) {
    // Step 1: Get destination address in PE's symmetric heap
    T *dest_actual = (T *)nvshmemi_device_state_d.peer_heap_base[pe] +
                     (dest - (T *)nvshmemi_device_state_d.heap_base);

    // Step 2: Check if we can use direct GPU-GPU access (P2P)
    if (nvshmemi_device_state_d.peer_heap_base_accessible[pe]) {
        // Path A: Direct memory copy via NVLink/PCIe
        for (size_t i = 0; i < nelems; i++) {
            dest_actual[i] = source[i];  // Direct store to remote GPU memory
        }
    } else {
        // Path B: Use transport (e.g., InfiniBand RDMA)
        nvshmemi_transfer_put_nbi<T>(dest, source, nelems, pe);
        nvshmemi_fence();  // Ensure completion
    }
}
```

**Location**: `src/include/non_abi/device/pt-to-pt/transfer_device.cuh` (template implementation)

#### 4. **Transport Layer** (if remote access needed)

```cpp
// src/include/non_abi/device/pt-to-pt/proxy_device.cuh
template <typename T>
__device__ void nvshmemi_transfer_put_nbi(T *dest, const T *source,
                                           size_t nelems, int pe) {
    // Use proxy channel to offload to CPU or network
    nvshmemi_proxy_rma_launcher(NVSHMEMI_OP_PUT, dest, source,
                                 nelems * sizeof(T), pe);
}
```

**Location**: `src/include/non_abi/device/pt-to-pt/proxy_device.cuh`

#### 5. **Proxy (GPU→Host Handoff)**

```cpp
// src/host/proxy/proxy.cpp
void nvshmemi_proxy_rma_launcher(...) {
    // GPU writes request to proxy channel
    // CPU proxy thread picks up request
    // Calls host-side transport
    transport->host_ops.rma_put(dest, source, size, pe);
}
```

**Location**: `src/host/proxy/proxy.cpp`

#### 6. **Host Transport Implementation**

```cpp
// src/host/comm/putget.cpp
int nvshmemi_put(void *dest, const void *source, size_t size, int pe) {
    transport_t *t = nvshmemi_state->transports[pe];

    // Select transport based on topology
    if (t->is_local) {
        // Local GPU: use CUDA memcpy
        cudaMemcpy(dest, source, size, cudaMemcpyDeviceToDevice);
    } else {
        // Remote GPU: use IB/UCX/etc.
        t->host_ops.rma(dest, source, size, pe, NVSHMEM_OP_PUT);
    }
}
```

**Location**: `src/host/comm/putget.cpp`

#### 7. **Actual Transport (e.g., InfiniBand)**

```cpp
// src/modules/transport/ibgda/ibgda.cpp
int ibgda_rma_put(void *dest, const void *source, size_t size, int pe) {
    struct ibv_send_wr wr = {0};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr.rdma.remote_addr = (uintptr_t)dest;
    wr.wr.rdma.rkey = remote_keys[pe];
    // ... setup scatter-gather list ...
    ibv_post_send(qp[pe], &wr, &bad_wr);  // Post RDMA write
}
```

**Location**: `src/modules/transport/ibgda/ibgda.cpp`

### Summary: C++ Call Stack

```
User Kernel
  └─> nvshmem_float_put()                    [device/nvshmem_defines.h:98]
       └─> nvshmemi_put<float>()              [non_abi/device/pt-to-pt/transfer_device.cuh]
            ├─> [P2P Path] Direct GPU store   [NVLink/PCIe hardware]
            └─> [Network Path]
                 └─> nvshmemi_proxy_launcher() [non_abi/device/pt-to-pt/proxy_device.cuh]
                      └─> proxy thread (CPU)   [host/proxy/proxy.cpp]
                           └─> transport_rma() [host/comm/putget.cpp]
                                └─> ibgda_put() [modules/transport/ibgda/ibgda.cpp]
                                     └─> ibv_post_send() [InfiniBand Verbs]
```

---

### Python API Call Path: `nvshmem.core.reduce()`

Now let's trace a **collective operation** from Python to C++:

#### 1. **User Code** (Python Application)

```python
# nvshmem4py/examples/on-stream.py
import nvshmem.core

full_sum = nvshmem.core.array((1,), dtype="int")
partial_sum = nvshmem.core.array((1,), dtype="int")
stream = dev.create_stream()

nvshmem.core.reduce(
    nvshmem.core.Teams.TEAM_WORLD,
    full_sum, partial_sum, "sum", stream=stream
)
```

**Location**: `nvshmem4py/examples/on-stream.py:64`

#### 2. **Python Core API**

```python
# nvshmem4py/nvshmem/core/collective.py
def reduce(team, dest, src, op, stream=None):
    """
    Perform a reduction operation across all PEs in a team.
    """
    # Validate inputs
    _validate_array(dest)
    _validate_array(src)

    # Get data type and size
    dtype = _get_nvshmem_dtype(src.dtype)
    nelem = src.size

    # Map operation string to enum
    op_enum = _get_reduce_op(op)  # "sum" -> NVSHMEM_SUM

    # Call Cython binding
    if stream is not None:
        status = bindings.reduce_on_stream(
            team._handle, dest._ptr, src._ptr,
            nelem, dtype, op_enum, stream._handle
        )
    else:
        status = bindings.reduce(
            team._handle, dest._ptr, src._ptr,
            nelem, dtype, op_enum
        )

    if status != 0:
        raise NvshmemError(f"Reduce failed with status {status}")
```

**Location**: `nvshmem4py/nvshmem/core/collective.py` (conceptual - actual implementation varies)

#### 3. **Cython Binding Layer**

```cython
# nvshmem4py/nvshmem/bindings/cynvshmem.pyx
cdef int reduce_on_stream(nvshmem_team_t team, void* dest, void* src,
                           size_t nelem, int dtype, int op,
                           cudaStream_t stream) except* nogil:
    # Direct call to C++ API
    if dtype == DTYPE_INT:
        return _nvshmem._nvshmemx_int_sum_reduce_on_stream(
            team, <int*>dest, <int*>src, nelem, stream
        )
    elif dtype == DTYPE_FLOAT:
        return _nvshmem._nvshmemx_float_sum_reduce_on_stream(
            team, <float*>dest, <float*>src, nelem, stream
        )
    # ... other types ...
```

**Location**: `nvshmem4py/nvshmem/bindings/cynvshmem.pyx`

#### 4. **C++ Host API Entry Point**

```cpp
// src/include/host/nvshmemx_coll_api.h
int nvshmemx_int_sum_reduce_on_stream(nvshmem_team_t team, int *dest,
                                       const int *source, size_t nelem,
                                       cudaStream_t stream) {
    return nvshmemi_reduce_on_stream<int>(team, dest, source, nelem,
                                          NVSHMEM_SUM, stream);
}
```

**Location**: `src/include/host/nvshmemx_coll_api.h`

#### 5. **Host Collective Implementation**

```cpp
// src/host/coll/rdxn/rdxn_on_stream.cpp
template <typename T>
int nvshmemi_reduce_on_stream(nvshmem_team_t team, T *dest, const T *source,
                               size_t nelem, int op, cudaStream_t stream) {
    // Step 1: Get team info
    nvshmemi_team_t *team_info = nvshmemi_team_pool[team];
    int n_pes = team_info->size;
    int my_pe = team_info->my_pe;

    // Step 2: Choose algorithm based on size/topology
    if (use_nvls_algorithm(team, nelem)) {
        // NVLS (NVLink Sharp) path for in-node reduction
        return nvshmemi_reduce_nvls<T>(team, dest, source, nelem, op, stream);
    } else if (use_ring_algorithm(nelem, n_pes)) {
        // Ring algorithm for distributed reduction
        return nvshmemi_reduce_ring<T>(team, dest, source, nelem, op, stream);
    } else {
        // Tree/recursive doubling algorithm
        return nvshmemi_reduce_tree<T>(team, dest, source, nelem, op, stream);
    }
}
```

**Location**: `src/host/coll/rdxn/rdxn_on_stream.cpp`

#### 6. **Device Kernel Launch** (for GPU-side collective)

```cpp
// src/host/coll/rdxn/rdxn_on_stream.cpp (continued)
int nvshmemi_reduce_ring(team, dest, source, nelem, op, stream) {
    // Allocate scratch space
    T *scratch = nvshmem_malloc(nelem * sizeof(T));

    // Launch device kernel
    dim3 blocks = calculate_blocks(nelem);
    dim3 threads = 256;

    nvshmemi_reduce_ring_kernel<T><<<blocks, threads, 0, stream>>>(
        dest, source, scratch, nelem, op,
        team->start_pe, team->stride, team->size
    );

    return 0;
}
```

**Location**: `src/host/coll/rdxn/rdxn_on_stream.cpp`

#### 7. **Device Collective Kernel**

```cuda
// src/include/non_abi/device/coll/reducescatter.cuh
template <typename T>
__global__ void nvshmemi_reduce_ring_kernel(T *dest, const T *source,
                                             T *scratch, size_t nelem, int op,
                                             int start_pe, int stride, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Copy local data to scratch
    if (tid < nelem) {
        scratch[tid] = source[tid];
    }
    __syncthreads();

    // Ring reduction: send/recv with neighbors
    for (int step = 1; step < n_pes; step++) {
        int send_pe = (my_pe + step) % n_pes;
        int recv_pe = (my_pe - step + n_pes) % n_pes;

        // Send my partial result
        nvshmem_putmem(scratch, scratch, nelem * sizeof(T), send_pe);

        // Wait for neighbor's data
        nvshmem_quiet();

        // Reduce with received data
        if (tid < nelem) {
            scratch[tid] = reduce_op(scratch[tid], dest[tid], op);
        }
        __syncthreads();
    }

    // Final result
    if (tid < nelem) {
        dest[tid] = scratch[tid];
    }
}
```

**Location**: `src/include/non_abi/device/coll/reducescatter.cuh`

### Summary: Python Call Stack

```
Python User Code
  └─> nvshmem.core.reduce()                         [nvshmem4py/nvshmem/core/collective.py]
       └─> bindings.reduce_on_stream()              [nvshmem4py/nvshmem/bindings/cynvshmem.pyx]
            └─> _nvshmemx_int_sum_reduce_on_stream() [Cython → C++]
                 └─> nvshmemx_int_sum_reduce_on_stream() [host/nvshmemx_coll_api.h]
                      └─> nvshmemi_reduce_on_stream<>()    [host/coll/rdxn/rdxn_on_stream.cpp]
                           ├─> [NVLS Path] nvls_reduce()   [GPU multicast hardware]
                           └─> [Ring Path]
                                └─> nvshmemi_reduce_ring_kernel<<<>>>() [device/coll/reducescatter.cuh]
                                     └─> nvshmem_putmem()   [device RMA operations]
```

---

## Core Subsystems

### 1. Initialization & Bootstrap

**Purpose**: Discover PEs, establish communication, allocate symmetric heap

#### Bootstrap Mechanisms

NVSHMEM supports multiple bootstrap methods for PE discovery:

| Method | File | Use Case |
|--------|------|----------|
| **MPI** | `src/modules/bootstrap/mpi/` | Use existing MPI job launcher |
| **PMI** | `src/modules/bootstrap/pmi/` | Slurm/PBS job schedulers |
| **PMIx** | `src/modules/bootstrap/pmix/` | Modern HPC launchers |
| **UID** | `src/modules/bootstrap/uid/` | Standalone with unique ID broadcast |

#### Initialization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User calls nvshmem_init()                                    │
│    [host/nvshmem_api.h:52]                                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│ 2. nvshmemi_init_thread()                                       │
│    [host/init/init.cu:500+]                                     │
│    ┌────────────────────────────────────────────────────────┐  │
│    │ a. Parse environment variables (NVSHMEM_*)             │  │
│    │ b. Initialize CUDA runtime                             │  │
│    │ c. Bootstrap PE discovery                              │  │
│    │ d. Topology discovery (GPU interconnect)               │  │
│    │ e. Create symmetric heap                               │  │
│    │ f. Initialize transports                               │  │
│    │ g. Create default team (TEAM_WORLD)                    │  │
│    │ h. Initialize device state                             │  │
│    └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Source Files**:
- `src/host/init/init.cu`: Main initialization logic
- `src/host/bootstrap/bootstrap.cpp`: Bootstrap dispatcher
- `src/modules/bootstrap/*/`: Bootstrap implementations

**Key Data Structures**:
```cpp
// Global state
nvshmemi_state_t *nvshmemi_state;  // Host-side state
nvshmemi_device_host_state_t nvshmemi_device_state;  // Shared host-device state

struct nvshmemi_state_t {
    int mype;               // My PE ID
    int npes;               // Total number of PEs
    void *heap_base;        // Symmetric heap base address
    size_t heap_size;       // Heap size
    transport_t **transports;  // Array of transports [npes]
    nvshmemi_team_t **teams;   // Team objects
    // ... many more fields
};
```

**Location**: `src/include/internal/host/nvshmemi_types.h`

### 2. Memory Management

**Symmetric Heap**: The heart of NVSHMEM's PGAS model

#### Heap Allocation

```
┌─────────────────────────────────────────────────────────────────┐
│ Each PE allocates symmetric heap at SAME virtual address:       │
│                                                                  │
│ PE 0: 0x7f8000000000 ┌──────────────────┐                      │
│ PE 1: 0x7f8000000000 │  Symmetric Heap  │ (1 GB default)       │
│ PE 2: 0x7f8000000000 │                  │                      │
│ PE 3: 0x7f8000000000 └──────────────────┘                      │
│                                                                  │
│ When PE 0 calls: ptr = nvshmem_malloc(1024);                   │
│ - PE 0 gets: 0x7f8000000000 + offset                           │
│ - PE 1 gets: 0x7f8000000000 + offset (same offset!)            │
│ - ...                                                            │
│                                                                  │
│ This allows PE 0 to write to PE 1's ptr using SAME ADDRESS!    │
└─────────────────────────────────────────────────────────────────┘
```

#### Allocation Flow

```cpp
// 1. User calls
void *ptr = nvshmem_malloc(size);  // [host/nvshmem_api.h:83]

// 2. Host implementation
void *nvshmem_malloc(size_t size) {  // [host/mem/mem.cpp:350+]
    // a. Allocate from symmetric heap using dlmalloc
    void *ptr = nvshmemi_symmetric_heap_malloc(size);

    // b. Register with transport for RDMA
    for (int pe = 0; pe < npes; pe++) {
        transport[pe]->register_mem(ptr, size);
    }

    // c. Exchange addresses with all PEs (collective)
    nvshmemi_barrier_all();  // Ensure all PEs have allocated

    return ptr;
}
```

**Key Files**:
- `src/host/mem/mem.cpp`: Memory allocation/free
- `src/host/mem/mem_heap.cpp`: Symmetric heap management
- `src/host/mem/dlmalloc.cpp`: Doug Lea's malloc (symmetric allocator)

**Python Memory Management**:

```python
# nvshmem4py/nvshmem/core/memory.py
def buffer(size):  # [memory.py:54]
    # Calls C++ nvshmem_malloc via Cython
    ptr = bindings.nvshmem_malloc(size)

    # Wraps in cuda.core.Buffer (DLPack-compatible)
    return Buffer(ptr, size, memory_resource=NvshmemResource(...))
```

### 3. Communication Operations

#### Remote Memory Access (RMA)

NVSHMEM provides **one-sided** RMA operations:

| Operation | Description | Host API | Device API |
|-----------|-------------|----------|------------|
| **PUT** | Write to remote PE | `nvshmem_put()` | `nvshmem_put()` |
| **GET** | Read from remote PE | `nvshmem_get()` | `nvshmem_get()` |
| **P** | Write single element | `nvshmem_p()` | `nvshmem_p()` |
| **G** | Read single element | `nvshmem_g()` | `nvshmem_g()` |

**Transport Selection**:

```
┌──────────────────────────────────────────────────────────────┐
│ RMA Operation: nvshmem_put(dest, src, size, pe)             │
└─────────────────────┬────────────────────────────────────────┘
                      │
        ┌─────────────▼────────────┐
        │ Is PE on same node?      │
        └──────┬──────────┬────────┘
         YES   │          │  NO
    ┌──────────▼──┐    ┌──▼────────────────┐
    │ Is NVLink   │    │ Use network       │
    │ connected?  │    │ transport:        │
    └──┬──────┬───┘    │ - IBGDA (IB+GDR) │
   YES │   NO │        │ - UCX            │
  ┌────▼──┐┌─▼──────┐ │ - libfabric      │
  │Direct ││ CUDA   │ └──────────────────┘
  │NVLink ││ P2P    │
  │Store  ││ memcpy │
  └───────┘└────────┘
```

**Implementation**:

```cpp
// src/host/comm/putget.cpp (~53KB - largest communication file)
int nvshmemi_put(void *dest, const void *source, size_t size, int pe) {
    transport_t *t = nvshmemi_state->transports[pe];

    // Fast path: GPU-accessible memory
    if (nvshmemi_state->peer_heap_base_accessible[pe]) {
        void *dest_actual = (char *)nvshmemi_state->peer_heap_base[pe] +
                           ((char *)dest - (char *)nvshmemi_state->heap_base);
        cudaMemcpy(dest_actual, source, size, cudaMemcpyDeviceToDevice);
        return 0;
    }

    // Network path: use transport
    return t->host_ops.rma(dest, source, size, pe, NVSHMEM_OP_PUT);
}
```

**Location**: `src/host/comm/putget.cpp`

#### Atomic Memory Operations (AMO)

NVSHMEM supports **atomic operations** on remote memory:

| Operation | Types | Description |
|-----------|-------|-------------|
| `fetch` | int, float, ... | Atomically read remote value |
| `set` | int, float, ... | Atomically write remote value |
| `compare_swap` | int, ... | Atomic compare-and-swap |
| `fetch_add` | int, float, ... | Atomic read-modify-write |
| `fetch_and/or/xor` | int bitwise types | Atomic bitwise operations |

**Implementation**:

```cpp
// src/host/comm/amo.cpp
template <typename T>
T nvshmem_atomic_fetch_add(T *dest, T value, int pe) {
    // Check if transport supports GPU atomics
    if (nvshmemi_state->transports[pe]->caps & NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS) {
        // Direct GPU atomic via NVLink
        T *dest_actual = get_remote_addr(dest, pe);
        return atomicAdd(dest_actual, value);  // GPU atomic instruction
    } else {
        // Use network transport's atomic support
        return transport->amo_fetch_add(dest, value, pe);
    }
}
```

**Location**: `src/host/comm/amo.cpp`

### 4. Collective Operations

NVSHMEM implements **collective communication** primitives:

| Collective | File | Algorithm |
|------------|------|-----------|
| **Barrier** | `src/host/coll/barrier/` | Dissemination, tree |
| **Broadcast** | `src/host/coll/broadcast/` | Binomial tree, ring |
| **Reduce** | `src/host/coll/rdxn/` | Recursive doubling, ring, NVLS |
| **AllReduce** | `src/host/coll/rdxn/` | Reduce + broadcast |
| **AllToAll** | `src/host/coll/alltoall/` | Pairwise exchange |

#### Example: Barrier Implementation

```cpp
// src/host/coll/barrier/barrier_on_stream.cpp
int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    nvshmemi_team_t *t = nvshmemi_team_pool[team];

    // Choose algorithm based on team size
    if (t->size <= 8) {
        // Dissemination barrier for small teams
        return nvshmemi_barrier_dissemination(team, stream);
    } else {
        // Tree barrier for large teams
        return nvshmemi_barrier_tree(team, stream);
    }
}

int nvshmemi_barrier_dissemination(team, stream) {
    int distance = 1;
    int npes = team->size;
    int mype = team->my_pe;

    while (distance < npes) {
        int peer = (mype + distance) % npes;

        // Increment remote PE's arrival counter
        nvshmemx_uint64_atomic_inc_on_stream(
            &team->sync_counter[mype], peer, stream
        );

        // Wait for peer to increment my counter
        nvshmemx_uint64_wait_until_on_stream(
            &team->sync_counter[peer], NVSHMEM_CMP_GE, distance, stream
        );

        distance *= 2;
    }
}
```

**Location**: `src/host/coll/barrier/barrier_on_stream.cpp`

### 5. Proxy System

**Problem**: GPU threads cannot directly issue network I/O

**Solution**: Proxy threads on CPU handle network operations for GPU

```
┌────────────────────────────────────────────────────────────────┐
│                     Proxy Architecture                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU Thread                    Proxy Thread (CPU)              │
│  ┌──────────┐                  ┌──────────────┐               │
│  │ Kernel   │                  │ Proxy Worker │               │
│  │          │                  │ (pthread)    │               │
│  └────┬─────┘                  └───────┬──────┘               │
│       │                                │                       │
│       │ 1. Write request               │                       │
│       │    to proxy queue              │                       │
│       ├───────────────────────────────►│                       │
│       │                                │ 2. Poll queue         │
│       │                                │                       │
│       │                                │ 3. Issue network op   │
│       │                                │    (IB RDMA write)    │
│       │                                │                       │
│       │ 4. Poll completion             │                       │
│       │◄───────────────────────────────┤ 5. Mark complete      │
│       │                                │                       │
│  ┌────▼─────┐                  ┌───────▼──────┐               │
│  │ Continue │                  │ Return to    │               │
│  └──────────┘                  │ polling      │               │
│                                 └──────────────┘               │
└────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```cpp
// src/host/proxy/proxy.cpp (~66KB - complex proxy logic)
struct proxy_request_t {
    int type;        // PUT, GET, AMO, etc.
    void *dest;      // Destination address
    void *source;    // Source address
    size_t size;     // Transfer size
    int pe;          // Target PE
    volatile int *complete;  // Completion flag
};

void *nvshmemi_proxy_thread(void *arg) {
    proxy_channel_t *channel = (proxy_channel_t *)arg;

    while (!channel->shutdown) {
        // Poll for requests from GPU
        proxy_request_t *req = proxy_channel_poll(channel);

        if (req) {
            // Execute request
            switch (req->type) {
                case PROXY_OP_PUT:
                    transport_put(req->dest, req->source, req->size, req->pe);
                    break;
                case PROXY_OP_GET:
                    transport_get(req->dest, req->source, req->size, req->pe);
                    break;
                // ... other operations
            }

            // Mark completion
            *req->complete = 1;
        }
    }
}
```

**Location**: `src/host/proxy/proxy.cpp`

### 6. Transport Layer

The transport layer provides a **pluggable interface** for different interconnects:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Transport Abstraction                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Common Transport Interface (transport.h)                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ struct transport_ops {                                     │ │
│  │   int (*rma_put)(void *dest, void *src, size_t, int pe);  │ │
│  │   int (*rma_get)(void *dest, void *src, size_t, int pe);  │ │
│  │   int (*amo_fetch_add)(void *dest, value, int pe);        │ │
│  │   int (*register_mem)(void *ptr, size_t);                 │ │
│  │   // ... more ops                                          │ │
│  │ };                                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Transport Implementations:                                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │ P2P        │ │ IBGDA      │ │ IBDEVX     │ │ UCX        │  │
│  │ (NVLink/   │ │ (IB GPUDirect│ (IB DevX) │ │ (Unified   │  │
│  │  PCIe)     │ │  Async)    │ │            │ │  Comm X)   │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│       ▲              ▲              ▲              ▲            │
│       │              │              │              │            │
│  ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    │
│  │ Direct  │    │ RDMA    │    │ RDMA    │    │ Various │    │
│  │ Memory  │    │ via IB  │    │ via IB  │    │ fabrics │    │
│  │ Access  │    │ Verbs   │    │ DevX    │    │         │    │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

**Transport Selection** (during initialization):

```cpp
// src/host/transport/transport.cpp
int nvshmemi_setup_transports() {
    for (int pe = 0; pe < npes; pe++) {
        // 1. Check if PE is on same node
        if (nvshmemi_pe_dist.same_node[pe]) {
            // Try P2P (NVLink/PCIe)
            if (can_use_p2p(pe)) {
                transports[pe] = p2p_transport_init(pe);
                continue;
            }
        }

        // 2. Try network transports
        if (nvshmem_options.IB_ENABLE_IBGDA) {
            transports[pe] = ibgda_transport_init(pe);
        } else if (nvshmem_options.UCX_ENABLE) {
            transports[pe] = ucx_transport_init(pe);
        } else {
            // Fallback to default
            transports[pe] = default_transport_init(pe);
        }
    }
}
```

**Location**: `src/host/transport/transport.cpp`

**Transport Capabilities**:

```cpp
// Transport capability flags
#define NVSHMEM_TRANSPORT_CAP_CPU_WRITE         (1 << 0)  // CPU can write
#define NVSHMEM_TRANSPORT_CAP_CPU_READ          (1 << 1)  // CPU can read
#define NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD        (1 << 2)  // GPU can load
#define NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST        (1 << 3)  // GPU can store
#define NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS   (1 << 4)  // GPU atomics
// ...
```

Each transport declares its capabilities, allowing NVSHMEM to select optimal paths.

---

## Multi-Device Synchronization

NVSHMEM provides multiple synchronization primitives for coordinating across PEs:

### 1. Fences and Quiet

```cpp
nvshmem_fence();   // Order PUT/GET operations (memory fence)
nvshmem_quiet();   // Wait for all pending ops to complete
```

**Purpose**:
- `fence()`: Ensures all prior RMA operations are visible to remote PEs
- `quiet()`: Ensures all prior RMA operations have completed

**Implementation**:

```cpp
// src/host/comm/fence.cpp
void nvshmem_fence() {
    // For each transport, issue fence operation
    for (int pe = 0; pe < npes; pe++) {
        if (nvshmemi_state->transports[pe]->ops.fence) {
            nvshmemi_state->transports[pe]->ops.fence(pe);
        }
    }

    // GPU memory fence
    __threadfence_system();
}

// src/host/comm/quiet.cpp
void nvshmem_quiet() {
    // Wait for all pending operations to complete
    for (int pe = 0; pe < npes; pe++) {
        transport_t *t = nvshmemi_state->transports[pe];
        if (t->ops.quiet) {
            t->ops.quiet(pe);  // Transport-specific quiet
        }
    }

    // Ensure GPU work is done
    cudaDeviceSynchronize();
}
```

### 2. Barriers

```cpp
nvshmem_barrier_all();             // Global barrier (all PEs)
nvshmem_barrier(team);             // Team barrier
nvshmemx_barrier_on_stream(team, stream);  // Asynchronous barrier
```

**Algorithms**:

#### Dissemination Barrier (for small teams)

```
Round 0: Each PE sends to PE (mype + 1) % npes
Round 1: Each PE sends to PE (mype + 2) % npes
Round 2: Each PE sends to PE (mype + 4) % npes
...
Round k: Each PE sends to PE (mype + 2^k) % npes

After log2(npes) rounds, all PEs have synchronized.
```

#### Tree Barrier (for large teams)

```
         PE 0 (root)
        /    \
      PE 1    PE 2
     /  \    /  \
   PE 3 PE 4 PE 5 PE 6

Phase 1 (up): Children send to parents
Phase 2 (down): Parents broadcast completion to children
```

**Stream-Based Barriers**:

```cpp
// Asynchronous barrier - does not block host
nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);

// Internally, uses CUDA stream callbacks or GPU-side spinning
__global__ void barrier_kernel() {
    // Each thread increments remote PE's counter
    for (int round = 0; round < log2(npes); round++) {
        int peer = (mype + (1 << round)) % npes;
        nvshmem_uint64_atomic_inc(&barrier_counter[mype], peer);
        nvshmem_uint64_wait_until(&barrier_counter[peer],
                                   NVSHMEM_CMP_GE, round + 1);
    }
}
```

### 3. Wait/Test Operations

```cpp
// Wait until remote value satisfies condition
nvshmem_uint64_wait_until(uint64_t *ptr, int cmp, uint64_t value);

// Test (non-blocking version)
int result = nvshmem_uint64_test(uint64_t *ptr, int cmp, uint64_t value);
```

**Implementation** (device-side):

```cpp
// src/include/non_abi/device/wait/nvshmemi_wait_until_apis.cuh
template <typename T>
__device__ void nvshmemi_wait_until(T *ptr, int cmp, T value) {
    // Busy-wait loop
    while (1) {
        T current = *ptr;  // Load from symmetric heap

        bool done = false;
        switch (cmp) {
            case NVSHMEM_CMP_EQ: done = (current == value); break;
            case NVSHMEM_CMP_NE: done = (current != value); break;
            case NVSHMEM_CMP_GT: done = (current > value); break;
            case NVSHMEM_CMP_GE: done = (current >= value); break;
            case NVSHMEM_CMP_LT: done = (current < value); break;
            case NVSHMEM_CMP_LE: done = (current <= value); break;
        }

        if (done) break;

        // Optional: yield to allow other warps to run
        __nanosleep(100);  // CUDA 12+
    }
}
```

### 4. Signaling Operations

```cpp
// Put with signal: notify remote PE after transfer completes
nvshmem_putmem_signal(dest, src, size, sig_addr, signal_value, sig_op, pe);
```

**Use Case**: Overlap communication with computation

```cpp
// Producer PE
nvshmem_putmem_signal(remote_buffer, local_data, size,
                       &flag, 1, NVSHMEM_SIGNAL_SET, consumer_pe);

// Consumer PE
nvshmem_uint64_wait_until(&flag, NVSHMEM_CMP_EQ, 1);
// Now remote_buffer is ready to use
```

---

## Code Organization

### Directory Structure (Detailed)

```
nvshmem/
├── src/                                    # Core NVSHMEM implementation
│   ├── include/                            # Public and internal headers
│   │   ├── host/                           # Host-side public API
│   │   │   ├── nvshmem_api.h               # Core host API
│   │   │   ├── nvshmemx_api.h              # Extended host API
│   │   │   └── nvshmemx_coll_api.h         # Collective operations
│   │   ├── device/                         # Device-side public API
│   │   │   ├── nvshmem_defines.h           # Core device API (inline)
│   │   │   └── nvshmemx_defines.h          # Extended device API
│   │   ├── non_abi/                        # Internal implementation headers
│   │   │   ├── device/                     # Device implementation
│   │   │   │   ├── pt-to-pt/               # Point-to-point transfers
│   │   │   │   │   ├── transfer_device.cuh  # Template implementations
│   │   │   │   │   ├── proxy_device.cuh    # Proxy channel device-side
│   │   │   │   │   └── ibgda_device.cuh    # IBGDA device-side
│   │   │   │   ├── coll/                   # Collective device kernels
│   │   │   │   ├── wait/                   # Wait/test operations
│   │   │   │   └── common/                 # Common device utilities
│   │   │   └── nvshmem_version.h           # Version info
│   │   ├── internal/                       # Internal host headers
│   │   │   ├── host/                       # Host internal
│   │   │   │   ├── nvshmem_internal.h      # Main internal header
│   │   │   │   ├── nvshmemi_types.h        # Internal types
│   │   │   │   ├── nvshmemi_team.h         # Team management
│   │   │   │   └── util.h                  # Utilities
│   │   │   ├── host_transport/             # Transport interfaces
│   │   │   │   └── transport.h             # Transport API
│   │   │   └── bootstrap_host/             # Bootstrap interfaces
│   │   │       └── nvshmemi_bootstrap.h    # Bootstrap API
│   │   └── device_host/                    # Shared host-device headers
│   │       ├── nvshmem_types.h             # Common types
│   │       └── nvshmem_common.cuh          # Common utilities
│   │
│   ├── host/                               # Host-side implementation
│   │   ├── init/                           # Initialization
│   │   │   ├── init.cu                     # Main init (CUDA-compiled)
│   │   │   ├── cudawrap.cpp                # CUDA runtime wrapper
│   │   │   └── nvmlwrap.cpp                # NVML wrapper
│   │   ├── mem/                            # Memory management
│   │   │   ├── mem.cpp                     # Core memory ops
│   │   │   ├── mem_heap.cpp                # Symmetric heap (~108KB)
│   │   │   └── dlmalloc.cpp                # Allocator
│   │   ├── comm/                           # Communication
│   │   │   ├── putget.cpp                  # PUT/GET (~53KB - largest)
│   │   │   ├── amo.cpp                     # Atomic operations
│   │   │   ├── sync.cpp                    # Synchronization
│   │   │   ├── fence.cpp                   # Fence operations
│   │   │   └── quiet.cpp                   # Quiet operations
│   │   ├── coll/                           # Collective operations
│   │   │   ├── barrier/                    # Barrier implementations
│   │   │   ├── broadcast/                  # Broadcast implementations
│   │   │   ├── rdxn/                       # Reduction implementations
│   │   │   ├── alltoall/                   # All-to-all
│   │   │   └── reducescatter/              # Reduce-scatter
│   │   ├── proxy/                          # Proxy system
│   │   │   └── proxy.cpp                   # Proxy implementation (~66KB)
│   │   ├── transport/                      # Transport dispatcher
│   │   │   ├── transport.cpp               # Transport manager
│   │   │   └── p2p/                        # P2P transport
│   │   ├── team/                           # Team management
│   │   ├── topo/                           # Topology detection
│   │   └── util/                           # Utilities
│   │
│   ├── device/                             # Device-side implementation
│   │   └── init/                           # Device initialization
│   │
│   └── modules/                            # Pluggable modules
│       ├── bootstrap/                      # Bootstrap implementations
│       │   ├── mpi/                        # MPI bootstrap
│       │   ├── pmi/                        # PMI bootstrap
│       │   ├── pmix/                       # PMIx bootstrap
│       │   └── uid/                        # Unique ID bootstrap
│       └── transport/                      # Transport implementations
│           ├── ibgda/                      # InfiniBand GPUDirect Async
│           ├── ibdevx/                     # InfiniBand DevX
│           ├── ibrc/                       # InfiniBand RC
│           ├── ucx/                        # UCX transport
│           ├── libfabric/                  # libFabric transport
│           └── common/                     # Common transport code
│
├── nvshmem4py/                             # Python bindings
│   ├── nvshmem/                            # Python package
│   │   ├── __init__.py                     # Package init
│   │   ├── bindings/                       # Cython bindings
│   │   │   ├── nvshmem.pyx                 # Main Cython interface
│   │   │   ├── cynvshmem.pyx               # Wrapper functions
│   │   │   ├── _internal/                  # Internal bindings
│   │   │   └── device/                     # Device bindings (Numba)
│   │   └── core/                           # Pure Python core
│   │       ├── init_fini.py                # Init/finalize
│   │       ├── memory.py                   # Memory management
│   │       ├── rma.py                      # RMA operations
│   │       ├── collective.py               # Collectives
│   │       ├── teams.py                    # Team management
│   │       ├── interop/                    # Framework integration
│   │       │   ├── torch.py                # PyTorch support
│   │       │   └── cupy.py                 # CuPy support
│   │       └── device/                     # Device-side Python
│   │           └── numba/                  # Numba JIT support
│   └── examples/                           # Python examples
│
├── examples/                               # C++ examples
│   ├── hello.cpp                           # Simple host example
│   ├── dev-guide-ring.cu                   # Ring pattern
│   ├── on-stream.cu                        # Stream operations
│   ├── put-block.cu                        # Block PUT example
│   └── collective-launch.cu                # Collective launch
│
└── perftest/                               # Performance tests
    ├── host/                               # Host-side tests
    ├── device/                             # Device-side tests
    └── common/                             # Common test code
```

### Key File Sizes (Complexity Indicators)

| File | Size | Significance |
|------|------|--------------|
| `src/host/mem/mem_heap.cpp` | ~108KB | Largest - complex heap management |
| `src/host/proxy/proxy.cpp` | ~66KB | Complex proxy system |
| `src/host/comm/putget.cpp` | ~53KB | Complex RMA logic |
| `src/host/init/init.cu` | ~40KB | Complex initialization |

---

## Key Data Structures

### 1. Global State

```cpp
// src/include/internal/host/nvshmemi_types.h

// Global host state (one per process)
struct nvshmemi_state_t {
    int mype;                          // My PE ID
    int npes;                          // Total number of PEs
    void *heap_base;                   // Symmetric heap base
    size_t heap_size;                  // Heap size
    void **peer_heap_base;             // Remote heap bases [npes]
    int *peer_heap_base_accessible;    // Accessibility flags [npes]

    transport_t **transports;          // Transport per PE [npes]
    nvshmemi_team_t **teams;           // Team objects

    // Proxy
    proxy_state_t *proxy;              // Proxy thread state

    // Bootstrap
    bootstrap_handle_t boot_handle;    // Bootstrap interface

    // Options
    nvshmem_options_t options;         // Configuration options

    // ... many more fields (~100 total)
};

extern nvshmemi_state_t *nvshmemi_state;  // Global instance
```

### 2. Device State (Shared with Host)

```cpp
// src/include/device_host/nvshmem_types.h

// State accessible from both host and device
struct nvshmemi_device_host_state_t {
    int mype;                          // My PE ID
    int npes;                          // Total PEs
    void *heap_base;                   // Symmetric heap base
    size_t heap_size;                  // Heap size

    void **peer_heap_base;             // Remote heaps [npes]
    int *peer_heap_base_accessible;    // P2P accessibility [npes]

    // Device-side proxy channel
    nvshmemi_proxy_channel_t *proxy_channel;

    // Transport state pointers
    void **transport_bitmap;           // Transport capability bitmaps

    // ... device-specific fields
};

// Constant memory on device
__constant__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
```

### 3. Transport Structure

```cpp
// src/include/internal/host_transport/transport.h

struct nvshmem_transport {
    char *name;                        // "IBGDA", "UCX", etc.
    int capabilities;                  // Capability flags

    // Host operations
    struct {
        int (*rma)(void *dest, void *src, size_t, int pe, int op);
        int (*amo)(void *dest, void *src, void *fetch, int op, int pe);
        int (*fence)(int pe);
        int (*quiet)(int pe);
        int (*get_mem_handle)(nvshmem_mem_handle_t *, void *, size_t);
        // ... more ops
    } host_ops;

    // Device operations (if supported)
    struct {
        void (*rma_device)(void *dest, void *src, size_t, int pe);
        // ... device ops
    } device_ops;

    void *state;                       // Transport-specific state
};
```

### 4. Team Structure

```cpp
// src/include/internal/host/nvshmemi_team.h

struct nvshmemi_team {
    int my_pe;                         // My PE within team
    int size;                          // Team size
    int start_pe;                      // First PE in team
    int stride;                        // PE stride

    // Synchronization state
    uint64_t *sync_counter;            // For barriers
    uint64_t *pWrk;                    // Collective work arrays
    long *pSync;                       // Sync arrays

    // Topology
    int psync_len;                     // pSync length

    // Parent/config
    nvshmem_team_config_t config;      // Team configuration
};

// Predefined teams
#define NVSHMEM_TEAM_WORLD    0        // All PEs
#define NVSHMEMX_TEAM_NODE    1        // PEs on same node
```

---

## Important Design Patterns

### 1. Template Metaprogramming for Type Dispatch

NVSHMEM uses **C++ templates** to generate type-specific functions:

```cpp
// src/include/device/nvshmem_defines.h

// Macro expands to generate functions for all types
#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(MACRO) \
    MACRO(float, float)                              \
    MACRO(double, double)                            \
    MACRO(int, int)                                  \
    MACRO(long, long)                                \
    // ... more types

// Define PUT for each type
#define NVSHMEMI_TYPENAME_PUT_IMPL(TYPENAME, TYPE)   \
    __device__ __forceinline__ void nvshmem_##TYPENAME##_put( \
        TYPE *dest, const TYPE *source, size_t nelems, int pe) { \
        nvshmemi_put<TYPE>(dest, source, nelems, pe); \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_IMPL

// Expands to:
// nvshmem_float_put(...)
// nvshmem_double_put(...)
// nvshmem_int_put(...)
// ...
```

This pattern avoids code duplication while providing type-safe APIs.

### 2. Capability-Based Transport Selection

Transports declare **capabilities**, and NVSHMEM selects operations based on them:

```cpp
// Capability flags
#define NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST      (1 << 3)   // GPU can store
#define NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS (1 << 4)   // GPU atomics

// Example: selecting atomic path
if (transport->caps & NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS) {
    // Direct GPU atomic
    return gpu_atomic_add(dest, value, pe);
} else {
    // Use host-based atomic
    return host_atomic_add(dest, value, pe);
}
```

This allows new transports to be added without modifying core logic.

### 3. Inline Device Functions for Performance

Device-side APIs are **always inlined** to eliminate function call overhead:

```cpp
#define NVSHMEMI_DEVICE_PREFIX __device__
#define NVSHMEMI_DEVICE_ALWAYS_INLINE __forceinline__

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE
void nvshmem_put(...) {
    // Implementation inlined into caller
}
```

This is critical for GPU performance, as function calls can be expensive.

### 4. Symmetric Address Translation

NVSHMEM uses a **simple offset calculation** for symmetric addresses:

```cpp
// Given: local pointer to symmetric heap object
// Want: remote PE's pointer to same object

void *remote_ptr = (char *)peer_heap_base[pe] +
                   ((char *)local_ptr - (char *)heap_base);

// Example:
// heap_base = 0x7f8000000000
// local_ptr = 0x7f8000001000  (offset = 0x1000)
// peer_heap_base[2] = 0x7f9000000000  (PE 2's heap)
// remote_ptr = 0x7f9000001000  (same offset!)
```

This works because **all PEs allocate symmetric heap at the same virtual address**.

### 5. Stream-Ordered Operations

Modern NVSHMEM uses **CUDA streams** for asynchronous operations:

```cpp
// Traditional (blocking)
nvshmem_barrier_all();
cudaDeviceSynchronize();

// Stream-ordered (non-blocking)
nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);
// Kernel launch can continue immediately
my_kernel<<<blocks, threads, 0, stream>>>();
```

**Implementation**:
- Uses CUDA stream callbacks
- Or launches GPU-side synchronization kernels
- Integrates with CUDA graphs

---

## Python-Specific Architecture

### Python → C++ Bridge

The Python API follows this path:

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Python User Code                                            │
│    import nvshmem.core                                         │
│    buf = nvshmem.core.buffer(1024)                            │
└─────────────────────────┬──────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────┐
│ 2. Python Core API (Pure Python)                               │
│    nvshmem4py/nvshmem/core/memory.py                          │
│    - Input validation                                          │
│    - Device selection                                          │
│    - Reference tracking                                        │
└─────────────────────────┬──────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────┐
│ 3. Cython Bindings                                             │
│    nvshmem4py/nvshmem/bindings/cynvshmem.pyx                  │
│    cdef void* nvshmem_malloc(size_t size):                    │
│        return _nvshmem._nvshmem_malloc(size)                  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────┐
│ 4. C++ Host API                                                │
│    void *nvshmem_malloc(size_t size)                          │
│    [src/host/mem/mem.cpp]                                     │
└────────────────────────────────────────────────────────────────┘
```

### Numba Integration (Device-Side Python)

For **GPU kernels in Python**, NVSHMEM provides Numba support:

```python
from numba import cuda
import nvshmem.core

@cuda.jit
def my_kernel(data, dest_pe):
    # Device-side NVSHMEM calls in Python!
    nvshmem.device.put(data, data, len(data), dest_pe)
    nvshmem.device.quiet()
```

**Implementation**:
- Numba compiles Python → LLVM IR
- NVSHMEM provides **bitcode library** (`.bc` file)
- Numba links NVSHMEM functions during JIT compilation

**Location**: `nvshmem4py/nvshmem/core/device/numba/`

---

## Conclusion

This document provides a comprehensive overview of NVSHMEM's architecture. For detailed call traces and annotated examples, see:

- [C++ Examples Walkthrough](./CPP_EXAMPLES_WALKTHROUGH.md)
- [Python Examples Walkthrough](./PYTHON_EXAMPLES_WALKTHROUGH.md)

### Key Takeaways

1. **Layered Design**: Clean separation between user API, host implementation, transport, and device code
2. **Pluggable Transports**: Abstract transport interface allows multiple backends
3. **Symmetric Heap**: Core abstraction enabling PGAS model
4. **Template Metaprogramming**: Type-generic APIs without runtime overhead
5. **Device Inlining**: Zero-overhead device-side calls
6. **Proxy System**: Bridges GPU-host gap for network I/O
7. **Python Integration**: Seamless Python → C++ via Cython + Numba for device code

NVSHMEM's architecture prioritizes **performance** (zero-copy, inline functions) while maintaining **flexibility** (pluggable transports, multiple bootstrap methods) and **usability** (stream-ordered ops, Python bindings).
