# NVSHMEM C++ Examples: Annotated Walkthroughs

> **Purpose**: This document provides detailed, annotated walkthroughs of NVSHMEM C++ examples, tracing the complete call path from user-facing APIs down to the lowest-level implementations (device kernels, transport layers, and hardware).
>
> **Format**: For each example, we provide:
> - Code overview and purpose
> - Line-by-line annotated source code with inline comments
> - Complete call stack traces with file locations
> - Visual diagrams of data flow
> - Links to implementation details

---

## Table of Contents

1. [Example 1: hello.cpp - Host-Side Initialization](#example-1-hellocpp---host-side-initialization)
2. [Example 2: dev-guide-ring.cu - Device-Side RMA](#example-2-dev-guide-ringcu---device-side-rma)
3. [Example 3: put-block.cu - Block-Level RMA](#example-3-put-blockcu---block-level-rma)
4. [Example 4: on-stream.cu - Stream-Based Collectives](#example-4-on-streamcu---stream-based-collectives)
5. [Example 5: ring-bcast.cu - Custom Broadcast](#example-5-ring-bcastcu---custom-broadcast)
6. [Example 6: collective-launch.cu - Collective Kernel Launch](#example-6-collective-launchcu---collective-kernel-launch)

---

## Example 1: hello.cpp - Host-Side Initialization

### Purpose

Demonstrates the minimal NVSHMEM host-side application:
- Initialize NVSHMEM library
- Query PE information
- Allocate symmetric memory
- Clean shutdown

### Source Code with Annotations

```cpp
// examples/hello.cpp

#include <stdio.h>
#include <unistd.h>
#include "nvshmem.h"              // [1] Main NVSHMEM header

int main(int argc, char **argv) {
    char hostname[256];

    int ret = gethostname(hostname, 256);
    if (ret < 0) {
        printf("Failed to get hostname\n");
        return 1;
    }

    printf("[%s][%ld] Starting up...\n", hostname, (long)getpid());

    // [2] ═══════════════════════════════════════════════════════════
    // INITIALIZATION: Bootstrap PE discovery, setup symmetric heap
    // ═══════════════════════════════════════════════════════════════
    nvshmem_init();               // LINE 22
    // CALL STACK:
    //   └─> nvshmem_init() [src/include/host/nvshmem_api.h:52]
    //        └─> nvshmemi_init_thread() [src/host/init/init.cu:500+]
    //             ├─> bootstrap_init() - Discover PEs
    //             ├─> topology_init() - Map GPU interconnects
    //             ├─> transport_init() - Initialize IB/UCX/P2P
    //             ├─> symmetric_heap_init() - Allocate heap
    //             └─> team_init() - Create TEAM_WORLD

    // [3] ═══════════════════════════════════════════════════════════
    // QUERY PE INFORMATION
    // ═══════════════════════════════════════════════════════════════
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);  // LINE 24
    // Returns local PE ID within node (0-7 for 8-GPU node)
    // CALL STACK:
    //   └─> nvshmem_team_my_pe() [src/host/team/team_internal.cpp]
    //        └─> returns nvshmemi_state->teams[team]->my_pe

    // [4] ═══════════════════════════════════════════════════════════
    // SET CUDA DEVICE
    // ═══════════════════════════════════════════════════════════════
    cudaSetDevice(mype_node);     // LINE 25
    // Each PE uses a different GPU (PE 0→GPU 0, PE 1→GPU 1, etc.)

    // [5] ═══════════════════════════════════════════════════════════
    // ALLOCATE SYMMETRIC MEMORY
    // ═══════════════════════════════════════════════════════════════
    void *ptr = nvshmem_malloc(1);  // LINE 26
    // CALL STACK:
    //   └─> nvshmem_malloc() [src/host/mem/mem.cpp:350+]
    //        ├─> nvshmemi_symmetric_heap_malloc() [src/host/mem/mem_heap.cpp]
    //        │    └─> dlmalloc(size) - Allocate from symmetric heap
    //        ├─> transport->register_mem() - Register for RDMA
    //        └─> nvshmemi_barrier_all() - Collective to sync all PEs

    // Why allocate 1 byte? This forces symmetric heap initialization
    // after CUDA device is set, ensuring heap is allocated on correct GPU

    // [6] ═══════════════════════════════════════════════════════════
    // QUERY AND PRINT PE INFO
    // ═══════════════════════════════════════════════════════════════
    printf("[%s][%ld] Hello from PE %d of %d\n", hostname, (long)getpid(),
           nvshmem_my_pe(),       // LINE 28 - Global PE ID
           nvshmem_n_pes());      // LINE 28 - Total number of PEs
    // CALL STACK:
    //   └─> nvshmem_my_pe() [inline in src/include/host/nvshmem_api.h:75]
    //        └─> returns nvshmemi_state->mype
    //   └─> nvshmem_n_pes() [inline]
    //        └─> returns nvshmemi_state->npes

    // [7] ═══════════════════════════════════════════════════════════
    // FINALIZE: Cleanup resources, shutdown transports
    // ═══════════════════════════════════════════════════════════════
    nvshmem_finalize();           // LINE 31
    // CALL STACK:
    //   └─> nvshmemi_finalize() [src/host/init/init.cu:1200+]
    //        ├─> transport_finalize() - Close network connections
    //        ├─> symmetric_heap_finalize() - Free symmetric memory
    //        ├─> team_finalize() - Destroy teams
    //        └─> bootstrap_finalize() - Shutdown bootstrap

    return 0;
}
```

**Location**: [examples/hello.cpp](../examples/hello.cpp)

### Complete Call Stack Trace

#### `nvshmem_init()` Deep Dive

```
User Code: nvshmem_init()
  │
  ├─> [src/include/host/nvshmem_api.h:52-58]
  │   inline function that calls:
  │
  └─> nvshmemi_init_thread(NVSHMEM_THREAD_SERIALIZED, &provided, 0, NULL, version)
       │
       └─> [src/host/init/init.cu:500-800] (Main initialization function)
            │
            ├─> 1. Environment Variable Parsing [init.cu:510+]
            │   └─> nvshmemi_options_init() - Parse NVSHMEM_* env vars
            │
            ├─> 2. CUDA Initialization [init.cu:550+]
            │   ├─> cudaGetDeviceCount()
            │   ├─> cudaGetDevice()
            │   └─> cudaDeviceGetAttribute() - Query GPU capabilities
            │
            ├─> 3. Bootstrap (PE Discovery) [init.cu:600+]
            │   └─> bootstrap_init(NVSHMEMX_INIT_AUTO, &boot_handle)
            │        │
            │        └─> [src/host/bootstrap/bootstrap.cpp:100+]
            │             ├─> Try PMI if available [modules/bootstrap/pmi/]
            │             ├─> Try MPI if available [modules/bootstrap/mpi/]
            │             ├─> Try PMIx if available [modules/bootstrap/pmix/]
            │             └─> Fall back to UID [modules/bootstrap/uid/]
            │
            ├─> 4. Topology Discovery [init.cu:650+]
            │   └─> nvshmemi_init_topo()
            │        │
            │        └─> [src/host/topo/topo.cpp:200+]
            │             ├─> Detect NVLink connectivity
            │             ├─> Detect PCIe topology
            │             ├─> Build connectivity matrix
            │             └─> Identify same-node PEs
            │
            ├─> 5. Symmetric Heap Allocation [init.cu:700+]
            │   └─> nvshmemi_symmetric_heap_init()
            │        │
            │        └─> [src/host/mem/mem_heap.cpp:50+]
            │             ├─> Calculate heap size (default 1GB)
            │             ├─> cudaMalloc(&heap_base, heap_size)
            │             ├─> Initialize dlmalloc allocator
            │             └─> Exchange heap addresses with all PEs
            │
            ├─> 6. Transport Initialization [init.cu:750+]
            │   └─> nvshmemi_setup_transports()
            │        │
            │        └─> [src/host/transport/transport.cpp:300+]
            │             For each PE:
            │             ├─> If same_node && NVLink:
            │             │    └─> p2p_transport_init() [src/host/transport/p2p/p2p.cpp]
            │             ├─> Else if IB enabled:
            │             │    └─> ibgda_transport_init() [src/modules/transport/ibgda/ibgda.cpp]
            │             └─> Else if UCX enabled:
            │                  └─> ucx_transport_init() [src/modules/transport/ucx/ucx.cpp]
            │
            ├─> 7. Team Initialization [init.cu:780+]
            │   └─> nvshmemi_team_init()
            │        └─> Create predefined teams:
            │             ├─> NVSHMEM_TEAM_WORLD (all PEs)
            │             └─> NVSHMEMX_TEAM_NODE (same-node PEs)
            │
            └─> 8. Device State Initialization [init.cu:800+]
                └─> nvshmemi_update_device_state()
                     ├─> Populate nvshmemi_device_state struct
                     ├─> Copy to device constant memory
                     └─> cudaMemcpyToSymbol(nvshmemi_device_state_d, ...)
```

**Key Source Files**:
- [src/include/host/nvshmem_api.h:52-58](../src/include/host/nvshmem_api.h#L52-L58) - API entry point
- [src/host/init/init.cu:500-800](../src/host/init/init.cu#L500-L800) - Main initialization
- [src/host/bootstrap/bootstrap.cpp](../src/host/bootstrap/bootstrap.cpp) - Bootstrap dispatcher
- [src/host/topo/topo.cpp](../src/host/topo/topo.cpp) - Topology detection
- [src/host/mem/mem_heap.cpp](../src/host/mem/mem_heap.cpp) - Symmetric heap management

### Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    hello.cpp Execution Flow                    │
└────────────────────────────────────────────────────────────────┘

Start
  │
  ├─> nvshmem_init()
  │    ┌──────────────────────────────────────────────────────┐
  │    │ Bootstrap: Discover PEs via PMI/MPI/UID              │
  │    │ Result: mype=2, npes=4                               │
  │    └──────────────────────────────────────────────────────┘
  │    ┌──────────────────────────────────────────────────────┐
  │    │ Topology: Detect GPU interconnects                   │
  │    │ PE 0-1: NVLink (same node)                           │
  │    │ PE 2-3: NVLink (same node)                           │
  │    │ PE 0↔2: InfiniBand (cross-node)                      │
  │    └──────────────────────────────────────────────────────┘
  │    ┌──────────────────────────────────────────────────────┐
  │    │ Symmetric Heap: Allocate 1GB on each PE             │
  │    │ PE 0: heap @ 0x7f8000000000                         │
  │    │ PE 1: heap @ 0x7f8000000000 (same VA!)              │
  │    │ PE 2: heap @ 0x7f8000000000                         │
  │    │ PE 3: heap @ 0x7f8000000000                         │
  │    └──────────────────────────────────────────────────────┘
  │
  ├─> nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)
  │    └─> Returns: 0 (first PE on this node)
  │
  ├─> cudaSetDevice(0)
  │    └─> This PE uses GPU 0
  │
  ├─> nvshmem_malloc(1)
  │    ┌──────────────────────────────────────────────────────┐
  │    │ Allocate 1 byte from symmetric heap                 │
  │    │ Returns: 0x7f8000000010 (offset in heap)            │
  │    │ Registered with transports for RDMA                 │
  │    └──────────────────────────────────────────────────────┘
  │
  ├─> nvshmem_my_pe() → 2
  ├─> nvshmem_n_pes() → 4
  │
  └─> nvshmem_finalize()
       ┌──────────────────────────────────────────────────────┐
       │ Cleanup: Close transports, free heap, shutdown       │
       └──────────────────────────────────────────────────────┘

End
```

---

## Example 2: dev-guide-ring.cu - Device-Side RMA

### Purpose

Demonstrates **device-side (GPU kernel) RMA operations**:
- Launch CUDA kernel
- Perform `nvshmem_int_p()` (single-element PUT) from device
- Implement ring communication pattern

### Source Code with Annotations

```cuda
// examples/dev-guide-ring.cu

#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// [1] ═══════════════════════════════════════════════════════════
// DEVICE KERNEL: Ring shift pattern
// Each PE sends its PE ID to the next PE in a ring topology
// ═══════════════════════════════════════════════════════════════
__global__ void simple_shift(int *destination) {
    // [2] Query PE information from device
    // These are INLINE functions that read from constant memory
    int mype = nvshmem_my_pe();    // LINE 19
    // CALL STACK:
    //   └─> nvshmem_my_pe() [src/include/device/nvshmem_defines.h:55-57]
    //        └─> return nvshmemi_device_state_d.mype;
    //             // nvshmemi_device_state_d is in __constant__ memory

    int npes = nvshmem_n_pes();    // LINE 20
    // CALL STACK:
    //   └─> nvshmem_n_pes() [src/include/device/nvshmem_defines.h:59-61]
    //        └─> return nvshmemi_device_state_d.npes;

    // [3] Calculate ring destination: (mype + 1) % npes
    // Creates circular pattern: 0→1, 1→2, 2→3, 3→0
    int peer = (mype + 1) % npes;  // LINE 21

    // [4] ═══════════════════════════════════════════════════════════
    // DEVICE-SIDE RMA: Write single integer to remote PE
    // ═══════════════════════════════════════════════════════════════
    nvshmem_int_p(destination, mype, peer);  // LINE 23
    // Write "mype" to destination[0] on PE "peer"
    //
    // CALL STACK:
    //   └─> nvshmem_int_p() [src/include/device/nvshmem_defines.h:80-84]
    //        └─> nvshmemi_p<int>(destination, mype, peer)
    //             │
    //             └─> [src/include/non_abi/device/pt-to-pt/transfer_device.cuh:100+]
    //                  │
    //                  ├─> 1. Translate address to remote PE's heap
    //                  │   int *dest_actual =
    //                  │     (int *)nvshmemi_device_state_d.peer_heap_base[peer] +
    //                  │     (destination - (int *)nvshmemi_device_state_d.heap_base);
    //                  │
    //                  ├─> 2. Check if P2P accessible (NVLink/PCIe)
    //                  │   if (nvshmemi_device_state_d.peer_heap_base_accessible[peer]) {
    //                  │       // Direct GPU store
    //                  │       *dest_actual = mype;  ← Direct NVLink write!
    //                  │   }
    //                  │
    //                  └─> 3. Else: Use proxy/transport
    //                      else {
    //                          nvshmemi_proxy_rma_p(destination, mype, peer);
    //                          // GPU→CPU handoff via proxy channel
    //                          // CPU issues IB RDMA or other transport
    //                      }
}

int main(void) {
    int mype_node, msg;
    cudaStream_t stream;

    // [5] ═══════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════
    nvshmem_init();               // LINE 30
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // [6] ═══════════════════════════════════════════════════════════
    // ALLOCATE SYMMETRIC MEMORY
    // ═══════════════════════════════════════════════════════════════
    int *destination = (int *)nvshmem_malloc(sizeof(int));  // LINE 35
    // Allocates sizeof(int) on each PE's symmetric heap
    // Returns same offset on all PEs, enabling symmetric addressing

    // [7] ═══════════════════════════════════════════════════════════
    // LAUNCH KERNEL
    // ═══════════════════════════════════════════════════════════════
    simple_shift<<<1, 1, 0, stream>>>(destination);  // LINE 37
    // Single thread, single block
    // Executes nvshmem_int_p() from GPU

    // [8] ═══════════════════════════════════════════════════════════
    // SYNCHRONIZATION: Barrier across all PEs
    // ═══════════════════════════════════════════════════════════════
    nvshmemx_barrier_all_on_stream(stream);  // LINE 38
    // CALL STACK:
    //   └─> nvshmemx_barrier_all_on_stream() [src/host/coll/barrier/barrier_on_stream.cpp:50+]
    //        ├─> Launch barrier kernel on stream
    //        └─> Kernel uses dissemination/tree algorithm
    //             └─> [src/include/non_abi/device/coll/barrier.cuh]
    //                  For each round:
    //                    ├─> nvshmem_uint64_atomic_inc(&counter[mype], peer)
    //                    └─> nvshmem_uint64_wait_until(&counter[peer], ...)

    // [9] ═══════════════════════════════════════════════════════════
    // COPY RESULT TO HOST
    // ═══════════════════════════════════════════════════════════════
    cudaMemcpyAsync(&msg, destination, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);  // LINE 39

    cudaStreamSynchronize(stream); // LINE 41
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);  // LINE 42

    // [10] ═══════════════════════════════════════════════════════════
    // CLEANUP
    // ═══════════════════════════════════════════════════════════════
    nvshmem_free(destination);    // LINE 44
    nvshmem_finalize();           // LINE 45
    return 0;
}
```

**Location**: [examples/dev-guide-ring.cu](../examples/dev-guide-ring.cu)

### Complete Call Stack Trace: `nvshmem_int_p()`

```
GPU Kernel: simple_shift<<<>>>
  │
  └─> nvshmem_int_p(destination, mype, peer)
       │
       ├─> [src/include/device/nvshmem_defines.h:80-84]
       │   __device__ __forceinline__ void nvshmem_int_p(...) {
       │       nvshmemi_p<int>(dest, value, pe);
       │   }
       │
       └─> nvshmemi_p<int>(destination, mype, peer)
            │
            └─> [src/include/non_abi/device/pt-to-pt/transfer_device.cuh:100-150]
                 │
                 ├─> Step 1: Calculate remote address
                 │   │
                 │   ├─> void *heap_base = nvshmemi_device_state_d.heap_base;
                 │   │   // Constant memory: base of symmetric heap
                 │   │
                 │   ├─> void *peer_base = nvshmemi_device_state_d.peer_heap_base[peer];
                 │   │   // Base of peer's heap (exchanged during init)
                 │   │
                 │   └─> int *dest_actual = (int *)peer_base +
                 │                          (destination - (int *)heap_base);
                 │       // Translate local symmetric address to remote address
                 │
                 ├─> Step 2: Check accessibility
                 │   │
                 │   └─> if (nvshmemi_device_state_d.peer_heap_base_accessible[peer])
                 │        // Flag set during init based on NVLink/PCIe connectivity
                 │        {
                 │            // ★★★ FAST PATH: Direct GPU-GPU access ★★★
                 │            *dest_actual = mype;  // Direct store instruction
                 │            // This becomes a single CUDA LD/ST instruction
                 │            // If NVLink: 300+ GB/s bandwidth
                 │            // If PCIe: ~16-32 GB/s bandwidth
                 │        }
                 │
                 └─> Step 3: Fallback (if not P2P accessible)
                     │
                     └─> else {
                             // ★★★ SLOW PATH: Proxy + Network transport ★★★
                             nvshmemi_proxy_rma_p_async(destination, mype, peer);
                             │
                             └─> [src/include/non_abi/device/pt-to-pt/proxy_device.cuh:50+]
                                  │
                                  ├─> 1. Allocate request in proxy channel queue
                                  │   proxy_request_t *req = proxy_channel_get_request();
                                  │   req->type = NVSHMEMI_OP_P;
                                  │   req->dest = destination;
                                  │   req->value = mype;
                                  │   req->pe = peer;
                                  │
                                  ├─> 2. Signal proxy thread (CPU)
                                  │   proxy_channel_submit(req);
                                  │
                                  └─> 3. Proxy thread wakes up
                                       └─> [src/host/proxy/proxy.cpp:200+]
                                            │
                                            ├─> Poll proxy channel queue
                                            │   while (req = proxy_channel_poll()) {
                                            │
                                            ├─> Execute request
                                            │   transport[peer]->rma_p(dest, value, peer);
                                            │   │
                                            │   └─> [src/modules/transport/ibgda/ibgda.cpp:500+]
                                            │        │
                                            │        ├─> Build IB work request
                                            │        │   struct ibv_send_wr wr;
                                            │        │   wr.opcode = IBV_WR_RDMA_WRITE;
                                            │        │   wr.wr.rdma.remote_addr = dest;
                                            │        │   wr.wr.rdma.rkey = rkeys[peer];
                                            │        │
                                            │        └─> Post to InfiniBand
                                            │            ibv_post_send(qp[peer], &wr, &bad_wr);
                                            │            // NIC handles RDMA write
                                            │
                                            └─> Mark completion
                                                req->complete = 1;
                                         }
                         }
```

### Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│               dev-guide-ring.cu: Ring Communication Pattern            │
└────────────────────────────────────────────────────────────────────────┘

Initial State (after nvshmem_malloc):
  PE 0: destination = [????]
  PE 1: destination = [????]
  PE 2: destination = [????]
  PE 3: destination = [????]

Kernel Execution: simple_shift<<<>>>
  ┌──────────────────────────────────────────────────────────────┐
  │ PE 0 (mype=0):                                               │
  │   peer = (0 + 1) % 4 = 1                                     │
  │   nvshmem_int_p(destination, 0, 1)  ────────┐                │
  │                                              │                │
  │ PE 1 (mype=1):                               │                │
  │   peer = (1 + 1) % 4 = 2                     ▼                │
  │   nvshmem_int_p(destination, 1, 2)  ──────► PE 1: [0] ◄────┐ │
  │                                              │               │ │
  │ PE 2 (mype=2):                               │               │ │
  │   peer = (2 + 1) % 4 = 3                     ▼               │ │
  │   nvshmem_int_p(destination, 2, 3)  ──────► PE 2: [1] ◄──┐  │ │
  │                                              │            │  │ │
  │ PE 3 (mype=3):                               │            │  │ │
  │   peer = (3 + 1) % 4 = 0                     ▼            │  │ │
  │   nvshmem_int_p(destination, 3, 0)  ──────► PE 3: [2] ◄┐ │  │ │
  │                                              │          │ │  │ │
  └──────────────────────────────────────────────┼──────────┼─┼──┼─┘
                                                 │          │ │  │
                                                 ▼          │ │  │
                                               PE 0: [3] ◄──┘ │  │
                                                    ▲         │  │
                                                    └─────────┘  │
                                                       ▲         │
                                                       └─────────┘

After Kernel + Barrier:
  PE 0: destination = [3]  ← Received from PE 3
  PE 1: destination = [0]  ← Received from PE 0
  PE 2: destination = [1]  ← Received from PE 1
  PE 3: destination = [2]  ← Received from PE 2

Ring shift complete! Each PE has its left neighbor's ID.
```

### Hardware-Level View (NVLink Path)

For **NVLink-connected GPUs** (e.g., DGX A100):

```
┌────────────────────────────────────────────────────────────────┐
│         Hardware Execution: nvshmem_int_p() via NVLink         │
└────────────────────────────────────────────────────────────────┘

GPU 0 Thread                           GPU 1 Thread
┌───────────┐                          ┌───────────┐
│ Kernel    │                          │ Kernel    │
│ simple_   │                          │ simple_   │
│ shift     │                          │ shift     │
└─────┬─────┘                          └───────────┘
      │
      │ nvshmem_int_p(dest, 0, peer=1)
      │
      ├─> dest_actual = peer_heap_base[1] + offset
      │   = 0x7f9000000010  (PE 1's heap address)
      │
      ├─> *dest_actual = 0;  // Store instruction
      │    ↓
      │   [CUDA ST instruction]
      │    ↓
      │   [GPU 0 L2 Cache]
      │    ↓
      │   [NVLink Interface]
      │    ↓
      │   ════════════════════════  NVLink 3.0: 600 GB/s
      │                            ════════════════════════
      │                             ↓
      │                          [GPU 1 NVLink Interface]
      │                             ↓
      │                          [GPU 1 L2 Cache]
      │                             ↓
      │                          [GPU 1 HBM @ 0x7f9000000010]
      │                             ↓
      │                          destination[0] = 0  ✓
      │
      └─> Done! (No CPU involvement, no network, pure GPU-GPU)
```

**Key Points**:
- **Zero-copy**: Data goes directly from GPU 0 → GPU 1
- **Low latency**: ~1-2 µs for small transfers
- **High bandwidth**: 300-600 GB/s (NVLink 3.0/4.0)
- **No CPU involvement**: Entire operation in GPU hardware

---

## Example 3: put-block.cu - Block-Level RMA

### Purpose

Demonstrates **block-level RMA** with `nvshmemx_float_put_block()`:
- Optimized for IB/network transports
- All threads in a block cooperate to transfer data
- More efficient than per-thread `nvshmem_put()`

### Source Code with Annotations

```cuda
// examples/put-block.cu (simplified, see full version in repo)

#include <stdio.h>
#include <assert.h>
#include "bootstrap_helper.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#define THREADS_PER_BLOCK 1024

// [1] ═══════════════════════════════════════════════════════════
// KERNEL: Block-level PUT operation
// Each block writes THREADS_PER_BLOCK elements to remote PE
// ═══════════════════════════════════════════════════════════════
__global__ void set_and_shift_kernel(float *send_data, float *recv_data,
                                      int num_elems, int mype, int npes) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // [2] Initialize send buffer
    if (thread_idx < num_elems) {
        send_data[thread_idx] = mype;  // Each PE writes its ID
    }

    int peer = (mype + 1) % npes;

    // [3] ═══════════════════════════════════════════════════════════
    // BLOCK PUT: All threads in block cooperate to transfer data
    // ═══════════════════════════════════════════════════════════════
    int block_offset = blockIdx.x * blockDim.x;
    nvshmemx_float_put_block(
        recv_data + block_offset,        // Remote destination
        send_data + block_offset,        // Local source
        min(blockDim.x, num_elems - block_offset),  // Number of elements
        peer                             // Target PE
    );  // LINE 48
    //
    // CALL STACK:
    //   └─> nvshmemx_float_put_block() [src/include/device/nvshmemx_defines.h:200+]
    //        └─> nvshmemi_put_block<float>(dest, src, nelems, pe)
    //             │
    //             └─> [src/include/non_abi/device/pt-to-pt/transfer_device.cuh:300+]
    //                  │
    //                  ├─> Mode 1: If P2P accessible (NVLink/PCIe)
    //                  │   // All threads cooperate to copy
    //                  │   int tid = threadIdx.x;
    //                  │   int nelems_per_thread = (nelems + blockDim.x - 1) / blockDim.x;
    //                  │
    //                  │   for (int i = tid; i < nelems; i += blockDim.x) {
    //                  │       dest_actual[i] = src[i];  // Coalesced stores
    //                  │   }
    //                  │   __syncthreads();  // Ensure all threads done
    //                  │
    //                  └─> Mode 2: If network transport
    //                      // Elect leader thread to issue single RMA
    //                      if (threadIdx.x == 0) {
    //                          nvshmemi_proxy_rma_put(dest, src, nelems * sizeof(float), pe);
    //                          // Single network message instead of 1024 tiny messages!
    //                      }
    //                      __syncthreads();  // All threads wait for completion
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;
    float *send_data, *recv_data;
    int num_elems = 8192;
    int num_blocks;

    // [4] Initialization
    nvshmem_init();                     // LINE 70
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);

    // [5] Allocate symmetric buffers
    send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);  // LINE 81
    recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);  // LINE 82

    // [6] Launch kernel
    num_blocks = num_elems / THREADS_PER_BLOCK;
    set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        send_data, recv_data, num_elems, mype, npes
    );  // LINE 88

    cudaDeviceSynchronize();  // LINE 91

    // [7] Validation
    float *host = new float[num_elems];
    cudaMemcpy(host, recv_data, num_elems * sizeof(float), cudaMemcpyDeviceToHost);
    int ref = (mype - 1 + npes) % npes;  // Expected: left neighbor's ID
    bool success = true;
    for (int i = 0; i < num_elems; ++i) {
        if (host[i] != ref) {
            printf("Error at %d of rank %d: %f\n", i, mype, host[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[%d of %d] run complete \n", mype, npes);
    }

    // [8] Cleanup
    nvshmem_free(send_data);
    nvshmem_free(recv_data);
    nvshmem_finalize();
    return 0;
}
```

**Location**: [examples/put-block.cu](../examples/put-block.cu)

### Block PUT vs. Regular PUT: Performance Comparison

```
┌────────────────────────────────────────────────────────────────┐
│         Comparison: nvshmem_put() vs. nvshmemx_put_block()     │
└────────────────────────────────────────────────────────────────┘

Scenario: 1024 threads transferring 1024 floats to remote PE via IB

──────────────────────────────────────────────────────────────────
Option 1: nvshmem_float_put() (per-thread)
──────────────────────────────────────────────────────────────────

__global__ void bad_example(float *dest, float *src, int pe) {
    int tid = threadIdx.x;
    nvshmem_float_put(&dest[tid], &src[tid], 1, pe);  // Each thread
}

Problem:
  - 1024 separate RMA operations
  - Each goes through proxy → CPU → IB
  - 1024 tiny IB messages (4 bytes each!)
  - IB message rate limited (~10M msgs/sec)
  - Latency: ~50 µs × 1024 = ~50 ms total

──────────────────────────────────────────────────────────────────
Option 2: nvshmemx_float_put_block() (block-level)
──────────────────────────────────────────────────────────────────

__global__ void good_example(float *dest, float *src, int pe) {
    nvshmemx_float_put_block(dest, src, 1024, pe);  // All threads
}

Optimization:
  - All threads participate, but issue SINGLE RMA
  - Thread 0 issues one 4KB message
  - Other threads help with data staging
  - Latency: ~50 µs for entire transfer
  - 1000× faster!

──────────────────────────────────────────────────────────────────
```

### Implementation Details: `nvshmemx_put_block()`

```cuda
// src/include/non_abi/device/pt-to-pt/transfer_device.cuh

template <typename T>
__device__ void nvshmemi_put_block(T *dest, const T *src, size_t nelems, int pe) {
    // [1] Get remote address
    T *dest_actual = get_remote_addr(dest, pe);

    // [2] Check if P2P accessible
    if (peer_heap_base_accessible[pe]) {
        // ★ P2P Path: All threads cooperate to copy
        int tid = threadIdx.x;
        int nthreads = blockDim.x;

        // Distribute work across threads
        for (int i = tid; i < nelems; i += nthreads) {
            dest_actual[i] = src[i];  // Coalesced memory access
        }

        // Ensure all threads finish before kernel continues
        __syncthreads();

    } else {
        // ★ Network Path: Single thread issues RMA
        if (threadIdx.x == 0) {
            // Only thread 0 in block issues the transfer
            nvshmemi_proxy_rma_put_nbi(dest, src, nelems * sizeof(T), pe);

            // [Optional] Wait for completion
            nvshmemi_quiet();  // Ensures RMA completes
        }

        // All threads wait for thread 0 to finish
        __syncthreads();
    }
}
```

**Key Benefits**:
1. **Coalesced Memory Access**: Threads access contiguous memory (GPU performance)
2. **Single Network Message**: Reduces latency and message rate overhead
3. **Synchronization**: `__syncthreads()` ensures consistency

---

## Example 4: on-stream.cu - Stream-Based Collectives

### Purpose

Demonstrates **asynchronous stream-based operations**:
- Non-blocking collectives (`_on_stream` variants)
- Overlap computation with communication
- CUDA stream integration

### Source Code with Annotations

```cuda
// examples/on-stream.cu

#include <stdio.h>
#include "bootstrap_helper.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#define THRESHOLD 42
#define CORRECTION 7

// [1] ═══════════════════════════════════════════════════════════
// KERNEL 1: Accumulate local values
// ═══════════════════════════════════════════════════════════════
__global__ void accumulate(int *input, int *partial_sum) {
    int index = threadIdx.x;
    if (0 == index) *partial_sum = 0;
    __syncthreads();
    atomicAdd(partial_sum, input[index]);  // Reduce within GPU
}

// [2] ═══════════════════════════════════════════════════════════
// KERNEL 2: Conditional correction based on global sum
// ═══════════════════════════════════════════════════════════════
__global__ void correct_accumulate(int *input, int *partial_sum, int *full_sum) {
    int index = threadIdx.x;

    // Read global sum from all PEs (result of reduce)
    if (*full_sum > THRESHOLD) {
        input[index] = input[index] - CORRECTION;
    }

    if (0 == index) *partial_sum = 0;
    __syncthreads();
    atomicAdd(partial_sum, input[index]);
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;
    int *input;
    int *partial_sum;
    int *full_sum;
    int input_nelems = 512;
    int to_all_nelems = 1;
    cudaStream_t stream;

    // [3] Initialization
    nvshmem_init();               // LINE 70
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);    // LINE 77

    // [4] Allocate symmetric memory
    input = (int *)nvshmem_malloc(sizeof(int) * input_nelems);  // LINE 79
    partial_sum = (int *)nvshmem_malloc(sizeof(int));           // LINE 80
    full_sum = (int *)nvshmem_malloc(sizeof(int));              // LINE 81

    // [5] ═══════════════════════════════════════════════════════════
    // STEP 1: Launch local accumulation kernel
    // ═══════════════════════════════════════════════════════════════
    accumulate<<<1, input_nelems, 0, stream>>>(input, partial_sum);  // LINE 83
    // Runs asynchronously on stream

    // [6] ═══════════════════════════════════════════════════════════
    // STEP 2: Global reduction (SUM across all PEs)
    // ═══════════════════════════════════════════════════════════════
    nvshmemx_int_sum_reduce_on_stream(
        NVSHMEM_TEAM_WORLD,      // All PEs participate
        full_sum,                 // Destination (result written here)
        partial_sum,              // Source (local partial sum)
        to_all_nelems,            // Number of elements (1)
        stream                    // CUDA stream for ordering
    );  // LINE 84-85
    //
    // CALL STACK:
    //   └─> nvshmemx_int_sum_reduce_on_stream() [src/host/coll/rdxn/rdxn_on_stream.cpp:100+]
    //        │
    //        ├─> Choose algorithm based on team size & topology:
    //        │   ├─> If NVLS available (Hopper+ GPUs, same node):
    //        │   │    └─> nvshmemi_reduce_nvls<int>(...)  [Multicast reduction]
    //        │   │
    //        │   ├─> Else if small team (≤8 PEs):
    //        │   │    └─> nvshmemi_reduce_ring<int>(...)  [Ring algorithm]
    //        │   │
    //        │   └─> Else (large distributed team):
    //        │        └─> nvshmemi_reduce_tree<int>(...)  [Tree algorithm]
    //        │
    //        └─> Launch reduction kernel on stream:
    //             nvshmemi_reduce_kernel<int><<<blocks, threads, 0, stream>>>(...)
    //             │
    //             └─> [src/include/non_abi/device/coll/reducescatter.cuh:50+]
    //                  Kernel performs multi-step reduction:
    //                  for (int step = 0; step < log2(npes); step++) {
    //                      int peer = compute_peer(mype, step);
    //
    //                      // Send my data to peer
    //                      nvshmem_int_put(scratch, local_sum, 1, peer);
    //                      nvshmem_quiet();
    //
    //                      // Reduce with received data
    //                      local_sum += scratch[0];  // SUM operation
    //                  }
    //                  full_sum[0] = local_sum;  // Write final result

    // [7] ═══════════════════════════════════════════════════════════
    // STEP 3: Conditional kernel based on global sum
    // ═══════════════════════════════════════════════════════════════
    correct_accumulate<<<1, input_nelems, 0, stream>>>(
        input, partial_sum, full_sum
    );  // LINE 86
    // Kernel reads full_sum (result of reduction)
    // Applies correction if needed

    // [8] ═══════════════════════════════════════════════════════════
    // SYNCHRONIZE: Wait for all stream operations to complete
    // ═══════════════════════════════════════════════════════════════
    cudaStreamSynchronize(stream);  // LINE 87

    printf("[%d of %d] run complete \n", mype, npes);

    // [9] Cleanup
    cudaStreamDestroy(stream);
    nvshmem_free(input);
    nvshmem_free(partial_sum);
    nvshmem_free(full_sum);
    nvshmem_finalize();

    return 0;
}
```

**Location**: [examples/on-stream.cu](../examples/on-stream.cu)

### Stream Ordering and Asynchrony

```
┌────────────────────────────────────────────────────────────────┐
│       on-stream.cu: Asynchronous Stream Execution Timeline     │
└────────────────────────────────────────────────────────────────┘

CUDA Stream Timeline:
────────────────────────────────────────────────────────────────►
  │
  ├─> accumulate<<<>>> (kernel)
  │    [Runs on GPU, ~10 µs]
  │    ▼
  │   ┌──────────────────────┐
  │   │ GPU computes local   │
  │   │ partial_sum          │
  │   └──────────────────────┘
  │                           ▼
  ├─> nvshmemx_int_sum_reduce_on_stream(...)
  │    [Launches reduction kernel on stream]
  │    ▼
  │   ┌──────────────────────────────────────────────────┐
  │   │ Reduction kernel (multi-step):                   │
  │   │ Step 0: Exchange with PE (mype ^ 1)              │
  │   │ Step 1: Exchange with PE (mype ^ 2)              │
  │   │ Step 2: Exchange with PE (mype ^ 4)              │
  │   │ ...                                              │
  │   │ Result: full_sum = global sum across all PEs    │
  │   └──────────────────────────────────────────────────┘
  │                           ▼
  ├─> correct_accumulate<<<>>> (kernel)
  │    [Reads full_sum, applies correction]
  │    ▼
  │   ┌──────────────────────┐
  │   │ GPU applies          │
  │   │ correction if needed │
  │   └──────────────────────┘
  │                           ▼
  └─> cudaStreamSynchronize(stream)
       [Host blocks here until all stream work completes]

Key Benefits:
  1. CPU never blocks until cudaStreamSynchronize()
  2. All operations ordered by stream dependencies
  3. GPU stays busy (no CPU-GPU synchronization between ops)
  4. Can overlap with other streams or host work
```

### Reduction Algorithms (Detail)

#### Ring Algorithm (Used for Small Teams)

```
┌────────────────────────────────────────────────────────────────┐
│                Ring Reduction Algorithm (4 PEs)                │
└────────────────────────────────────────────────────────────────┘

Initial State:
  PE 0: partial_sum = 10
  PE 1: partial_sum = 20
  PE 2: partial_sum = 30
  PE 3: partial_sum = 40

Step 0: Each PE sends to (mype + 1) % 4
  PE 0 ─(10)→ PE 1
  PE 1 ─(20)→ PE 2
  PE 2 ─(30)→ PE 3
  PE 3 ─(40)→ PE 0

  PE 0: local = 10, received 40 → new = 50
  PE 1: local = 20, received 10 → new = 30
  PE 2: local = 30, received 20 → new = 50
  PE 3: local = 40, received 30 → new = 70

Step 1: Send updated values
  PE 0 ─(50)→ PE 1
  PE 1 ─(30)→ PE 2
  PE 2 ─(50)→ PE 3
  PE 3 ─(70)→ PE 0

  PE 0: local = 50, received 70 → new = 120 (skip, not final)
  ...

After log2(4) = 2 steps:
  All PEs have full_sum = 100 (10 + 20 + 30 + 40)
```

---

## Example 5: ring-bcast.cu - Custom Broadcast

### Purpose

Demonstrates **custom collective implementation** using NVSHMEM primitives:
- Signaling operations for synchronization
- Ring-based broadcast algorithm
- `nvshmemx_collective_launch()` for coordinated kernel execution

### Source Code with Annotations

```cuda
// examples/ring-bcast.cu

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// [1] ═══════════════════════════════════════════════════════════
// KERNEL: Ring broadcast implementation
// Root PE sends data around a ring to all other PEs
// ═══════════════════════════════════════════════════════════════
__global__ void ring_bcast(int *data, size_t nelem, int root, uint64_t *psync) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;  // Next PE in ring

    // [2] ═══════════════════════════════════════════════════════════
    // ROOT: Initiate broadcast by setting flag
    // ═══════════════════════════════════════════════════════════════
    if (mype == root) {
        *psync = 1;  // Signal: "I have valid data"
    }  // LINE 24

    // [3] ═══════════════════════════════════════════════════════════
    // ALL PEs: Wait for data to arrive (wait for signal)
    // ═══════════════════════════════════════════════════════════════
    nvshmem_signal_wait_until(psync, NVSHMEM_CMP_NE, 0);  // LINE 26
    //
    // CALL STACK:
    //   └─> nvshmem_signal_wait_until() [src/include/device/nvshmem_defines.h:500+]
    //        └─> nvshmemi_wait_until<uint64_t>(psync, CMP_NE, 0)
    //             │
    //             └─> [src/include/non_abi/device/wait/nvshmemi_wait_until_apis.cuh:50+]
    //                  // Busy-wait loop
    //                  while (1) {
    //                      uint64_t current = *psync;  // Load from symmetric heap
    //                      if (current != 0) break;    // Condition satisfied
    //                      __nanosleep(100);           // Optional: yield to other warps
    //                  }
    //
    // This blocks until:
    //   - Root sets psync = 1 (immediately for root PE)
    //   - OR neighbor sends signal after receiving data

    // [4] ═══════════════════════════════════════════════════════════
    // LAST PE in ring: Don't forward (ring complete)
    // ═══════════════════════════════════════════════════════════════
    if (mype == npes - 1) return;  // LINE 28

    // [5] ═══════════════════════════════════════════════════════════
    // FORWARD DATA to next PE in ring
    // ═══════════════════════════════════════════════════════════════
    nvshmem_int_put(data, data, nelem, peer);  // LINE 30
    // Copy my data to peer's data buffer

    // [6] ═══════════════════════════════════════════════════════════
    // ENSURE PUT completes before signaling
    // ═══════════════════════════════════════════════════════════════
    nvshmem_fence();  // LINE 31
    //
    // CALL STACK:
    //   └─> nvshmem_fence() [src/include/device/nvshmem_defines.h:600+]
    //        └─> __threadfence_system()  // GPU memory fence
    //             // Ensures all prior stores (PUT) are visible to other PEs

    // [7] ═══════════════════════════════════════════════════════════
    // SIGNAL next PE: "Your data is ready!"
    // ═══════════════════════════════════════════════════════════════
    nvshmemx_signal_op(psync, 1, NVSHMEM_SIGNAL_SET, peer);  // LINE 32
    //
    // CALL STACK:
    //   └─> nvshmemx_signal_op() [src/include/device/nvshmemx_defines.h:300+]
    //        └─> nvshmemi_signal_op<uint64_t>(psync, 1, SET, peer)
    //             │
    //             └─> Remote write: psync@peer = 1
    //                  // Atomically sets peer's psync to 1
    //                  // Peer's wait_until() will now exit

    // [8] Reset local flag for next collective
    *psync = 0;  // LINE 34
}

int main(void) {
    size_t data_len = 32;
    cudaStream_t stream;

    // [9] Initialization
    nvshmem_init();               // LINE 41
    int mype = nvshmem_my_pe();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // [10] Allocate symmetric memory
    int *data = (int *)nvshmem_malloc(sizeof(int) * data_len);      // LINE 49
    int *data_h = (int *)malloc(sizeof(int) * data_len);            // LINE 50
    uint64_t *psync = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));  // LINE 51

    // [11] Initialize data (only root has real data initially)
    for (size_t i = 0; i < data_len; i++) {
        data_h[i] = mype + i;  // Each PE initializes differently
    }
    cudaMemcpyAsync(data, data_h, sizeof(int) * data_len,
                    cudaMemcpyHostToDevice, stream);

    // [12] ═══════════════════════════════════════════════════════════
    // PRE-BARRIER: Ensure all PEs ready before broadcast
    // ═══════════════════════════════════════════════════════════════
    int root = 0;
    nvshmemx_barrier_all_on_stream(stream);  // LINE 61

    // [13] ═══════════════════════════════════════════════════════════
    // COLLECTIVE LAUNCH: Coordinate kernel across all PEs
    // ═══════════════════════════════════════════════════════════════
    dim3 gridDim(1), blockDim(1);
    void *args[] = {&data, &data_len, &root, &psync};

    nvshmemx_collective_launch(
        (const void *)ring_bcast,  // Kernel function
        gridDim, blockDim,          // Grid/block dimensions
        args,                       // Kernel arguments
        0,                          // Shared memory size
        stream                      // CUDA stream
    );  // LINE 62
    //
    // CALL STACK:
    //   └─> nvshmemx_collective_launch() [src/host/device/launch/collective_launch.cpp:50+]
    //        ├─> Step 1: Barrier (ensure all PEs ready)
    //        │    └─> nvshmem_barrier_all()
    //        │
    //        ├─> Step 2: All PEs launch kernel simultaneously
    //        │    └─> cudaLaunchKernel(ring_bcast, gridDim, blockDim, args, 0, stream)
    //        │
    //        └─> Step 3: Barrier (ensure all kernels started)
    //             └─> nvshmem_barrier_all()
    //
    // Ensures synchronized start across all PEs

    // [14] ═══════════════════════════════════════════════════════════
    // POST-BARRIER: Ensure broadcast completes before proceeding
    // ═══════════════════════════════════════════════════════════════
    nvshmemx_barrier_all_on_stream(stream);  // LINE 63

    // [15] Copy result to host and verify
    cudaMemcpyAsync(data_h, data, sizeof(int) * data_len,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < data_len; i++) {
        if (data_h[i] != (int)i) {  // Should have root's data (0 + i)
            printf("PE %d error, data[%zu] = %d expected data[%zu] = %d\n",
                   mype, i, data_h[i], i, (int)i);
        }
    }

    // [16] Cleanup
    nvshmem_free(data);
    nvshmem_free(psync);
    free(data_h);
    nvshmem_finalize();
    return 0;
}
```

**Location**: [examples/ring-bcast.cu](../examples/ring-bcast.cu)

### Ring Broadcast Execution Flow

```
┌────────────────────────────────────────────────────────────────┐
│            Ring Broadcast: 4 PEs, Root = PE 0                  │
└────────────────────────────────────────────────────────────────┘

Time    PE 0 (root)      PE 1            PE 2            PE 3
────────────────────────────────────────────────────────────────
T0      data = [0..31]  data = [???]    data = [???]    data = [???]
        psync = 0       psync = 0       psync = 0       psync = 0

T1      psync = 1       wait(psync≠0)   wait(psync≠0)   wait(psync≠0)
        ↓ (signal set)  ↓ (waiting...)  ↓ (waiting...)  ↓ (waiting...)
        PUT data → PE1  │               │               │
        FENCE           │               │               │
        SIGNAL psync@1  │               │               │
         └──────────────►│               │               │

T2      done            psync = 1       wait(psync≠0)   wait(psync≠0)
                        data = [0..31]  ↓ (waiting...)  ↓ (waiting...)
                        PUT data → PE2  │               │
                        FENCE           │               │
                        SIGNAL psync@2  │               │
                         └──────────────►│               │

T3      done            done            psync = 1       wait(psync≠0)
                                        data = [0..31]  ↓ (waiting...)
                                        PUT data → PE3  │
                                        FENCE           │
                                        SIGNAL psync@3  │
                                         └──────────────►│

T4      done            done            done            psync = 1
                                                        data = [0..31]
                                                        (last PE, return)

Final State:
  All PEs have data = [0..31] (root's data)
  Broadcast complete!
```

### Key Techniques

1. **Signaling for Synchronization**:
   - `nvshmem_signal_wait_until()`: Busy-wait on flag
   - `nvshmemx_signal_op()`: Atomically set remote flag

2. **Ordering with Fence**:
   - `nvshmem_fence()`: Ensures PUT visible before signal
   - Prevents race: signal arrives before data

3. **Collective Launch**:
   - `nvshmemx_collective_launch()`: Synchronized kernel start across PEs
   - Avoids deadlock from unsynchronized kernel launches

---

## Example 6: collective-launch.cu - Collective Kernel Launch

### Purpose

Demonstrates **`nvshmemx_collective_launch()`**:
- Synchronized kernel execution across all PEs
- Avoids deadlocks in kernels with global synchronization
- Ring reduction example

### Source Code with Annotations

```cuda
// examples/collective-launch.cu

#include <stdio.h>
#include "bootstrap_helper.h"
#include "nvshmem.h"
#include "nvshmemx.h"

// [1] ═══════════════════════════════════════════════════════════
// KERNEL: Ring reduction (sum values across all PEs)
// ═══════════════════════════════════════════════════════════════
__global__ void reduce_ring(int *target, int mype, int npes) {
    int peer = (mype + 1) % npes;
    int lvalue = mype;  // Start with my PE ID

    // [2] Ring algorithm: npes iterations
    for (int i = 0; i < npes; i++) {
        // [3] Send my value to next PE in ring
        nvshmem_int_p(target, lvalue, peer);  // LINE 45

        // [4] ═══════════════════════════════════════════════════════
        // GLOBAL BARRIER: All PEs must reach this point
        // ═══════════════════════════════════════════════════════════
        nvshmem_barrier_all();  // LINE 46
        //
        // CALL STACK:
        //   └─> nvshmem_barrier_all() [src/include/device/nvshmem_defines.h:700+]
        //        └─> nvshmemi_barrier_all_device()
        //             │
        //             └─> [src/include/non_abi/device/coll/barrier.cuh:50+]
        //                  Dissemination barrier:
        //                  for (int d = 1; d < npes; d *= 2) {
        //                      int peer = (mype + d) % npes;
        //
        //                      // Increment peer's arrival counter
        //                      nvshmem_uint64_atomic_inc(&sync_arr[mype], peer);
        //
        //                      // Wait for peer to increment my counter
        //                      nvshmem_uint64_wait_until(&sync_arr[peer],
        //                                                 NVSHMEM_CMP_GE, round);
        //                  }
        //
        // ★ CRITICAL: If PEs don't launch kernel simultaneously,
        //             this barrier will DEADLOCK!
        //             That's why we use nvshmemx_collective_launch()

        // [5] Read value sent by previous PE
        lvalue = *target + mype;  // LINE 47

        // [6] Another barrier before next iteration
        nvshmem_barrier_all();  // LINE 48
    }

    // [7] Write final result
    *target = lvalue;  // LINE 51
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;

    // [8] Initialization
    nvshmem_init();               // LINE 69
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);

    // [9] Allocate symmetric memory
    int *u = (int *)nvshmem_calloc(1, sizeof(int));  // Device memory // LINE 78
    int *h = (int *)calloc(1, sizeof(int));           // Host memory   // LINE 79

    // [10] ═══════════════════════════════════════════════════════════
    // COLLECTIVE LAUNCH: Synchronized kernel start
    // ═══════════════════════════════════════════════════════════════
    void *args[] = {&u, &mype, &npes};
    dim3 dimBlock(1);
    dim3 dimGrid(1);

    nvshmemx_collective_launch(
        (const void *)reduce_ring,  // Kernel
        dimGrid, dimBlock,           // Dimensions
        args,                        // Arguments
        0,                           // Shared memory
        0                            // Default stream
    );  // LINE 85-86
    //
    // CALL STACK:
    //   └─> nvshmemx_collective_launch() [src/host/device/launch/collective_launch.cpp:100+]
    //        │
    //        ├─> Step 1: PRE-LAUNCH BARRIER
    //        │    └─> nvshmem_barrier_all()  [src/host/coll/barrier/barrier.cpp]
    //        │         // Ensures all PEs reach this point before launching
    //        │         // Uses CPU-side barrier (fast, ~µs)
    //        │
    //        ├─> Step 2: LAUNCH KERNEL
    //        │    └─> cudaLaunchKernel(kernel, grid, block, args, smem, stream)
    //        │         // Each PE launches kernel on its own GPU
    //        │         // After barrier, launches are nearly simultaneous
    //        │
    //        └─> Step 3: POST-LAUNCH BARRIER
    //             └─> nvshmem_barrier_all()
    //                  // Ensures all kernels started before returning
    //                  // Prevents host from racing ahead
    //
    // Why needed?
    //   - Without collective launch, PE 0 might start kernel at T=0
    //   - PE 3 might start kernel at T=1000µs (due to MPI delays, etc.)
    //   - When PE 0 reaches nvshmem_barrier_all() in kernel, PE 3 hasn't started
    //   - Result: DEADLOCK!
    //
    // With collective launch:
    //   - All PEs launch within ~µs of each other
    //   - GPU kernels all reach barriers at approximately same time
    //   - No deadlock!

    cudaDeviceSynchronize();  // LINE 87
    // Wait for kernel to complete

    // [11] Copy result to host
    cudaMemcpy(h, u, sizeof(int), cudaMemcpyDeviceToHost);
    printf("results on device [%d] is %d \n", mype, h[0]);

    // Expected result for 4 PEs:
    //   Initial: lvalue = mype
    //   Iter 0: lvalue = target + mype = (mype-1) + mype (from neighbor)
    //   ...
    //   After npes iterations: sum of all contributions

    // [12] Cleanup
    nvshmem_free(u);
    free(h);
    nvshmem_finalize();
    return 0;
}
```

**Location**: [examples/collective-launch.cu](../examples/collective-launch.cu)

### Why Collective Launch is Necessary

```
┌────────────────────────────────────────────────────────────────┐
│          Problem: Regular Launch vs. Collective Launch         │
└────────────────────────────────────────────────────────────────┘

──────────────────────────────────────────────────────────────────
Scenario: Regular cudaLaunchKernel() (WITHOUT collective launch)
──────────────────────────────────────────────────────────────────

CPU Timeline:
  PE 0: nvshmem_init() → kernel<<<>>> (T=1000µs)
  PE 1: nvshmem_init() → kernel<<<>>> (T=1050µs)  ← 50µs delay
  PE 2: nvshmem_init() → kernel<<<>>> (T=1100µs)  ← 100µs delay
  PE 3: nvshmem_init() → kernel<<<>>> (T=5000µs)  ← 4000µs delay!
        (PE 3 delayed due to slow MPI, network, etc.)

GPU Kernel Timeline:
  T=1000µs: PE 0 kernel starts
            Reaches nvshmem_barrier_all()
            Waiting for PE 1, 2, 3...

  T=1050µs: PE 1 kernel starts
            Reaches barrier
            Waiting...

  T=1100µs: PE 2 kernel starts
            Reaches barrier
            Waiting...

  T=5000µs: PE 3 kernel FINALLY starts  ← 4000µs later!
            Reaches barrier
            All PEs now in barrier → proceed

Problem:
  - PEs 0-2 wasted 4000µs spinning in barrier
  - If PE 3 never starts (crash, bug), DEADLOCK forever!

──────────────────────────────────────────────────────────────────
Solution: nvshmemx_collective_launch()
──────────────────────────────────────────────────────────────────

CPU Timeline:
  PE 0: nvshmem_init() → collective_launch() at T=5000µs
         ↓ Pre-launch barrier (waits for all PEs)
  PE 1: nvshmem_init() → collective_launch() at T=5000µs
         ↓
  PE 2: nvshmem_init() → collective_launch() at T=5000µs
         ↓
  PE 3: nvshmem_init() → collective_launch() at T=5000µs
         ↓ (slowest PE)

  T=5000µs: All PEs release from barrier simultaneously
            kernel<<<>>> on PE 0
            kernel<<<>>> on PE 1
            kernel<<<>>> on PE 2
            kernel<<<>>> on PE 3
            ↓ All kernels start within ~µs

GPU Kernel Timeline:
  T=5001µs: All kernels executing
  T=5002µs: All reach nvshmem_barrier_all() nearly simultaneously
            → No wasted waiting!

Benefit:
  - Eliminates waiting time
  - Prevents deadlock
  - Simplifies debugging (deterministic start time)
```

---

## Summary: Key Patterns in C++ Examples

### 1. Initialization Pattern

```cpp
nvshmem_init();                              // Bootstrap & setup
int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
cudaSetDevice(mype_node);                    // Select GPU
void *ptr = nvshmem_malloc(size);            // Allocate symmetric memory
```

### 2. Device RMA Pattern

```cuda
__global__ void kernel() {
    int peer = compute_peer();
    nvshmem_TYPE_put(dest, src, nelems, peer);  // One-sided RMA
    nvshmem_fence();                             // Order operations
    nvshmem_quiet();                             // Wait for completion
}
```

### 3. Stream-Based Collective Pattern

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

kernel_1<<<grid, block, 0, stream>>>();
nvshmemx_collective_on_stream(team, ..., stream);
kernel_2<<<grid, block, 0, stream>>>();

cudaStreamSynchronize(stream);  // Block host
```

### 4. Synchronization Pattern

```cuda
__global__ void kernel() {
    // Signal-based
    nvshmem_put(...);
    nvshmem_fence();
    nvshmemx_signal_op(flag, value, op, peer);
    nvshmem_wait_until(flag, cmp, value);

    // Barrier-based
    nvshmem_barrier_all();
}
```

### 5. Cleanup Pattern

```cpp
nvshmem_free(symmetric_ptr);
nvshmem_finalize();  // Shutdown transports, free resources
```

---

## Cross-Reference to Architecture.md

For detailed explanations of the components referenced in these examples, see:

- **Initialization**: [Architecture.md § Core Subsystems → Initialization](./ARCHITECTURE.md#1-initialization--bootstrap)
- **Memory Management**: [Architecture.md § Core Subsystems → Memory Management](./ARCHITECTURE.md#2-memory-management)
- **RMA Operations**: [Architecture.md § Core Subsystems → Communication Operations](./ARCHITECTURE.md#3-communication-operations)
- **Collective Operations**: [Architecture.md § Core Subsystems → Collective Operations](./ARCHITECTURE.md#4-collective-operations)
- **Synchronization**: [Architecture.md § Multi-Device Synchronization](./ARCHITECTURE.md#multi-device-synchronization)
- **Transport Layer**: [Architecture.md § Core Subsystems → Transport Layer](./ARCHITECTURE.md#6-transport-layer)

---

## Next Steps

For Python examples, see [PYTHON_EXAMPLES_WALKTHROUGH.md](./PYTHON_EXAMPLES_WALKTHROUGH.md).
