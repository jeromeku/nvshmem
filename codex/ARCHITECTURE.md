**NVSHMEM Architecture**

Purpose
- Explain NVSHMEM internals for developers: end‑to‑end call paths, layering, transports, and synchronization.
- Cover both C++ and Python APIs all the way to host/device implementations and kernels.

Scope
- NVSHMEM 3.x code in this repo
- C++ host/device APIs, transports, bootstrap, symmetric heap
- Python nvshmem4py bindings and core wrappers

Overview
- NVSHMEM is a PGAS library for multi‑GPU systems that provides GPU‑initiated and host‑initiated one‑sided RMA, atomics, and collectives over symmetric memory.
- Targets intra‑node (NVLink/NVSwitch, PCIe) and inter‑node (e.g., Libfabric/EFA) via pluggable transports.

Architecture Layers

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Applications                                                                  │
│  - C++/CUDA apps (examples/*.cu, *.cpp)                                       │
│  - Python apps (nvshmem4py/examples/*.py)                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Public APIs                                                                    │
│  - C++ host headers: nvshmem.h, nvshmemx.h (host + on‑stream)                 │
│  - C++ device headers: device inline APIs (nvshmem_defines.h)                 │
│  - Python: nvshmem4py core + cython bindings                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Host Runtime                                                                   │
│  - Initialization/bootstrap, teams, symmetric heap, proxy, barriers           │
│  - RMA/AMO/collective host entry points                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ GPU Entry (on‑stream paths)                                                    │
│  - Host launches small CUDA kernels that call device transfer templates       │
├──────────────────────────────────────────────────────────────────────────────┤
│ Device Runtime                                                                 │
│  - Device inline APIs → transfer templates → (IBGDA or GPU→Host proxy)        │
│  - Device collectives, signal/wait, teams                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Transports                                                                     │
│  - P2P (cudaMemcpyAsync), Libfabric/EFA backends, etc.                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

Key Components (code map)
- Public headers
  - C++ umbrella headers: [nvshmem.h:14](src/include/nvshmem.h#L14), [nvshmemx.h:16](src/include/nvshmemx.h#L16)
  - Host API declarations: [nvshmem_api.h:14](src/include/host/nvshmem_api.h#L14), [nvshmemx_api.h:20](src/include/host/nvshmemx_api.h#L20)
  - Device API inlines: [nvshmem_defines.h:17](src/include/device/nvshmem_defines.h#L17)
- Initialization & state
  - Device+host shared types/state: [nvshmem_types.h:1](src/include/device_host/nvshmem_types.h#L1)
  - Host state and transports: [nvshmemi_types.h:29](src/include/internal/host/nvshmemi_types.h#L29)
  - Host init/main flow: [init.cu:994](src/host/init/init.cu#L994) (nvshmemi_common_init)
  - Device init entry: [init_device.cu:139](src/device/init/init_device.cu#L139) (nvshmemi_init_thread)
- RMA/Sync on host
  - Put/Get host entry: [putget.cpp:364](src/host/comm/putget.cpp#L364)
  - Signal/wait and quiet: [sync.cpp:60](src/host/comm/sync.cpp#L60), [quiet.cpp:80](src/host/comm/quiet.cpp#L80)
- On‑stream GPU entry (host‑launched kernels)
  - RMA entrypoints: [rma.cu:11](src/host/comm/rma.cu#L11) → [nvshmemi_h_to_d_rma_defs.cuh:44](src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh#L44)
  - Barrier/sync kernels: [barrier.cu:13](src/host/stream/coll/barrier/barrier.cu#L13) → [nvshmemi_h_to_d_coll_defs.cuh:44](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L44)
- Device transfer paths
  - Device inlines → transfer templates: [nvshmem_defines.h:81](src/include/device/nvshmem_defines.h#L81), [transfer_device.cuh.in:104](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L104)
  - GPU proxy channel: [proxy_device.cuh:1](src/include/non_abi/device/pt-to-pt/proxy_device.cuh#L1)
- Transports
  - Transport interfaces: [transport.h:119](src/include/internal/host_transport/transport.h#L119)
  - Libfabric implementation: [libfabric.cpp:740](src/modules/transport/libfabric/libfabric.cpp#L740)
- Python
  - High‑level APIs: nvshmem4py/nvshmem/core/*.py
  - Cython bindings: [nvshmem.pyx:1397](nvshmem4py/nvshmem/bindings/nvshmem.pyx#L1397), [cynvshmem.pyx:1](nvshmem4py/nvshmem/bindings/cynvshmem.pyx#L1)

Initialization & Symmetric Heap

- C++ host apps call `nvshmem_init()` which wraps `nvshmemi_init_thread()`:
  - [nvshmem_api.h:52](src/include/host/nvshmem_api.h#L52) (nvshmem_init)
  - [init_device.cu:139](src/device/init/init_device.cu#L139) (nvshmemi_init_thread)
    - Calls hostlib init to bootstrap and register device state callbacks: [init.cu:1220](src/host/init/init.cu#L1220)
    - If fully initialized, sets up device‑only state and resolves CUDA device
  - Host library bootstrap resolves environment and bootstrap method (MPI/UID):
    - [init.cu:1302](src/host/init/init.cu#L1302) (nvshmemx_hostlib_init_attr)
    - `src/host/init/init.cu:994` (nvshmemi_common_init) → builds transport map, sets up connections, symmetric heap, teams, and proxy.
    - Symmetric heap allocation backend lives in `src/host/mem/mem_heap.cpp` (heap classes, VMM/egm paths). Public API:
      - `src/host/mem/mem_heap.cpp:2288` (nvshmem_malloc) calls `nvshmemi_check_state_and_init()` then allocates from heap and performs barrier‑all.
      - `src/host/init/init.cu:1195` (nvshmemi_check_state_and_init)

Python init flows mirror C++:
- `nvshmem.core.init(...)` selects MPI/UID mode, sets up `InitAttr`, then calls bindings:
  - `nvshmem4py/nvshmem/core/init_fini.py:112` (init)
  - `nvshmem4py/nvshmem/bindings/nvshmem.pyx:1397` (hostlib_init_attr) → `src/host/init/init.cu:1302`
  - Finalize via `nvshmem4py/nvshmem/bindings/nvshmem.pyx:1403` (hostlib_finalize) → `src/host/init/init.cu:1434`

RMA Data Paths (Host‑Initiated)

Two major host‑side paths exist for put/get:

1) Off‑stream host path (CPU issues data movement)
- API wrappers funnel to a single helper that selects transport, maps symmetric pointers, and posts RMA:
  - `src/host/comm/putget.cpp:232` (nvshmemi_prepare_and_post_rma)
- Intra‑node mapped heap (peer mapped): fast path uses cudaMemcpyAsync on internal streams:
  - `src/host/comm/putget.cpp:200` (nvshmemi_p2p_rma_optimized)
- Registered path (unmapped/remote): also uses cudaMemcpyAsync for staged copies, then optional signal op:
  - [putget.cpp:120](src/host/comm/putget.cpp#L120) (nvshmemi_p2p_rma_registered)
- Direct transport call for single‑word `P` and chunked multi‑send for bulk:
  - [nvshmem_internal.h:168](src/include/internal/host/nvshmem_internal.h#L168) (nvshmemi_process_multisend_rma)
  - [transport.h:119](src/include/internal/host_transport/transport.h#L119) (RMA signature)
  - Libfabric RMA implementation: [libfabric.cpp:740](src/modules/transport/libfabric/libfabric.cpp#L740)

2) On‑stream host path (GPU executes the transfer)
- Host composes device kernel args and launches a tiny proxy kernel on the user stream:
  - [rma.cu:11](src/host/comm/rma.cu#L11) (nvshmemi_proxy_rma_launcher)
  - [nvshmemi_h_to_d_rma_defs.cuh:44](src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh#L44) (proxy entrypoints)
- Those kernels call device transfer templates, which pick between IBGDA (GPU‑NIC) or GPU→Host proxy channel:
  - [transfer_device.cuh.in:104](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L104) (nvshmemi_transfer_rma<>)
  - Proxy channel details: [proxy_device.cuh:198](src/include/non_abi/device/pt-to-pt/proxy_device.cuh#L198)

RMA & Signals (Representative APIs)
- Typed/size‑generic host puts call the same helper:
  - [putget.cpp:364](src/host/comm/putget.cpp#L364) (nvshmem_putmem)
  - [putget.cpp:375](src/host/comm/putget.cpp#L375) (nvshmemx_putmem_on_stream)
  - [putget.cpp:511](src/host/comm/putget.cpp#L511) (nvshmemx_putmem_signal_on_stream)
- Device API inlines expand to transfer templates and signal ops:
  - [nvshmem_defines.h:81](src/include/device/nvshmem_defines.h#L81) (nvshmem_TYPENAME_p)
  - [nvshmem_defines.h:107](src/include/device/nvshmem_defines.h#L107) (nvshmem_TYPENAME_put_signal)

Transports (Host)

- Selection and posting
  - The helper selects `nvshmemi_state->selected_transport_for_rma[pe]` and obtains memory handles for local/remote symmetric memory or registered buffers:
    - [nvshmem_internal.h:130](src/include/internal/host/nvshmem_internal.h#L130) (get_local_mem_handle)
    - [nvshmem_internal.h:152](src/include/internal/host/nvshmem_internal.h#L152) (get_remote_mem_handle)
- Libfabric path (example)
  - `src/modules/transport/libfabric/libfabric.cpp:740` (nvshmemt_libfabric_rma)
  - Handles P (single element), PUT/GET, and optionally write‑with‑immediate for signaling.

Device Data Paths

- Device inlines in `nvshmem_defines.h` forward to internal transfer templates:
  - `src/include/device/nvshmem_defines.h:81` (nvshmem_TYPENAME_p)
  - `src/include/device/nvshmem_defines.h:206` (nvshmem_putmem)
- Transfer selection
  - `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:56` (nvshmemi_transfer_syncapi_update_mem)
  - `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:104` (nvshmemi_transfer_rma<SCOPE,op>)
    - IBGDA path if available, otherwise GPU→Host proxy channel ops (`nvshmemi_proxy_rma_nbi`, `nvshmemi_proxy_quiet`).
  - Signals on device: `src/include/device/nvshmem_defines.h:578` (signal fetch/wait), and put_signal templates in the same header.

Synchronization & Ordering

- Host quiet/fence
  - `src/host/comm/quiet.cpp:23` (nvshmem_quiet) drains internal streams; per‑transport quiet via host_ops; on‑stream quiet may launch a small device kernel to perform `threadfence_system` or remote‑transport quiet.
  - `src/host/comm/quiet.cpp:80` (nvshmemx_quiet_on_stream)
- Signals/Waits
  - On‑stream wait uses CUDA stream waits if available, else launches a device wait kernel:
    - `src/host/comm/sync.cpp:60` (nvshmemx_signal_wait_until_on_stream)
- Barriers/Sync
  - Host entry points launch threadgroup device kernels which implement dissemination/NVL algorithms:
    - `src/host/coll/barrier/barrier.cpp:20` (nvshmem_barrier) → `src/host/stream/coll/barrier/barrier.cu:13` (nvshmemi_call_barrier_on_stream_kernel)
    - Device side: `src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:44` (barrier_on_stream_kernel_threadgroup)
    - Algorithms: `src/include/non_abi/device/coll/barrier.cuh:196` (nvshmemi_sync_algo_threadgroup)

Teams & State

- Device/host shared state records heap, channels, connectivity and team pool:
  - `src/include/device_host/nvshmem_types.h:211` (nvshmemi_device_host_state_t fields)
  - Host state and transport map: `src/include/internal/host/nvshmemi_types.h:29`
- Team API visible to host and device:
  - Host declarations: `src/include/host/nvshmem_api.h:589`
  - Device wrappers: `src/include/device/nvshmem_defines.h:26` (nvshmem_team_my_pe, etc.)

Python Stack (nvshmem4py)

- High‑level user APIs (nvshmem.core)
  - Initialization/finalization and module/library init: `nvshmem4py/nvshmem/core/init_fini.py:112`, `nvshmem4py/nvshmem/core/init_fini.py:214`
  - Symmetric memory wrapping CUDA Buffers (tracked per device): `nvshmem4py/nvshmem/core/nvshmem_types.py:117` (NvshmemResource)
  - Array/tensor interop layers: `nvshmem4py/nvshmem/core/interop/cupy.py:56`, `nvshmem4py/nvshmem/core/interop/torch.py`
  - Host‑initiated collectives and RMA helpers: `nvshmem4py/nvshmem/core/collective.py:28`, `nvshmem4py/nvshmem/core/rma.py:76`
- Cython bindings
  - Host control and collectives map directly onto `nvshmemx_*_on_stream` entry points declared in `.pxd` and implemented in `.pyx`:
    - `nvshmem4py/nvshmem/bindings/nvshmem.pxd:241` (put/get/signal/collectives) → `nvshmem4py/nvshmem/bindings/nvshmem.pyx`
    - Underlying functions resolve to C++ host API in `src/include/host/nvshmemx_api.h` and its definitions in `src/host/*`.
- Device kernels from Python (Numba)
  - `nvshmem4py/nvshmem/bindings/device/numba/entry_point.h:1` exposes nvshmem device APIs to NVRTC for Numba kernels; helper loader wires include paths so `#include <nvshmem.h>` works during NVRTC.

Host vs Device End‑to‑End: Annotated Call Examples

1) Host‑initiated put with signal on a CUDA stream
- User (C++) calls nvshmemx_putmem_signal_on_stream:
  - [putget.cpp:511](src/host/comm/putget.cpp#L511) → helper [putget.cpp:232](src/host/comm/putget.cpp#L232)
  - Mapped peer → `cudaMemcpyAsync` + on‑stream signal: [sync.cpp:102](src/host/comm/sync.cpp#L102)
  - Remote peer → device proxy kernel: [rma.cu:11](src/host/comm/rma.cu#L11) → [nvshmemi_h_to_d_rma_defs.cuh:75](src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh#L75)
  - Device template executes put+signal: [transfer_device.cuh.in:137](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L137)
  - IBGDA/GPU→Host proxy channel do the actual movement: [proxy_device.cuh:206](src/include/non_abi/device/pt-to-pt/proxy_device.cuh#L206)

2) Device‑initiated nvshmem_float_put in a kernel
- Device inline: [nvshmem_defines.h:98](src/include/device/nvshmem_defines.h#L98) → `nvshmemi_put<T,SCOPE>`
- IBGDA vs proxy channel: [transfer_device.cuh.in:104](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L104)
  - Proxy encodes and queues requests: [proxy_device.cuh:206](src/include/non_abi/device/pt-to-pt/proxy_device.cuh#L206)

3) Python nvshmem.core.reduce(..., stream=...)
- High‑level resolves dtype/op then calls binding:
  - [collective.py:110](nvshmem4py/nvshmem/core/collective.py#L110) → `bindings.<dtype>_<op>_reduce_on_stream`
- Host launches device collective kernel:
  - [nvshmemi_h_to_d_coll_defs.cuh:44](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L44)

Bootstrap & Finalization (C++ and Python)
- C++ app
  - `src/include/host/nvshmem_api.h:52` (nvshmem_init) → `src/device/init/init_device.cu:139` (nvshmemi_init_thread)
  - Host bootstrap: `src/host/init/init.cu:1220` (nvshmemid_hostlib_init_attr) → `src/host/init/init.cu:994` (nvshmemi_common_init)
  - `src/device/init/init_device.cu:171` (nvshmemi_finalize) called by `src/include/host/nvshmem_api.h:70` (nvshmem_finalize)
- Python app
  - `nvshmem4py/nvshmem/core/init_fini.py:112` (init) → `nvshmem4py/nvshmem/bindings/nvshmem.pyx:1397` (hostlib_init_attr)
  - `nvshmem4py/nvshmem/core/init_fini.py:214` (finalize) → `nvshmem4py/nvshmem/bindings/nvshmem.pyx:1403` (hostlib_finalize)

Memory Model & Symmetry
- Symmetric heap allocated on all PEs at the same virtual address; visible on device via `nvshmemi_device_state_d.heap_base` and per‑peer bases.
  - Host allocations: `src/host/mem/mem_heap.cpp:2288` (nvshmem_malloc)
  - Symmetric pointer translation in host fast paths: `src/host/comm/putget.cpp:205` (NVSHMEMU_MAPPED_PTR_TRANSLATE)
  - Peer mapping/hints: `src/include/internal/host/nvshmem_internal.h:130` (handle lookup)

Multi‑Device Synchronization
- Ordering and completion via `quiet` and `fence` on host and device; device collectives implement threadgroup sync.
- Wait/Signal on CPU stream uses CUDA stream memory ops if available:
  - `src/host/comm/sync.cpp:60` (nvshmemx_signal_wait_until_on_stream)
  - Device wait/signal templates: `src/include/host/nvshmem_api.h:335+` (device decls) and inlines in `nvshmem_defines.h`.
- Barriers:
  - Host: `src/host/coll/barrier/barrier.cpp:20` → launch kernel `src/host/stream/coll/barrier/barrier.cu:13`
  - Device: dissemination/barrier templates `src/include/non_abi/device/coll/barrier.cuh:196`

Code Organization (selected)
- Public headers: `src/include/*`
- Host runtime: `src/host/*`
- Device runtime: `src/device/*`, `src/include/non_abi/device/*`
- Transports: `src/modules/transport/*`
- Examples: `examples/*`, `nvshmem4py/examples/*`
- Python: `nvshmem4py/nvshmem/*`

Troubleshooting Pointers
- Undefined behavior when calling APIs before init: guarded by `nvshmemi_check_state_and_init()` `src/host/init/init.cu:1195`
- On‑stream ops on remote transports require proxy kernels; missing transport or unsupported combos trigger explicit errors in `nvshmemi_prepare_and_post_rma` `src/host/comm/putget.cpp:232`
- Version checks between host/device libs: `src/host/init/init.cu:1226`
