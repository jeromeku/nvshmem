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
  - C++ umbrella headers: `src/include/nvshmem.h:14`, `src/include/nvshmemx.h:16`
  - Host API declarations: `src/include/host/nvshmem_api.h:14`, `src/include/host/nvshmemx_api.h:20`
  - Device API inlines: `src/include/device/nvshmem_defines.h:17`
- Initialization & state
  - Device+host shared types/state: `src/include/device_host/nvshmem_types.h:1`
  - Host state and transports: `src/include/internal/host/nvshmemi_types.h:29`
  - Host init/main flow: `src/host/init/init.cu:994` (nvshmemi_common_init)
  - Device init entry: `src/device/init/init_device.cu:139` (nvshmemi_init_thread)
- RMA/Sync on host
  - Put/Get host entry: `src/host/comm/putget.cpp:364` (nvshmem_putmem, typed wrappers nearby)
  - Signal/wait and quiet: `src/host/comm/sync.cpp:60`, `src/host/comm/quiet.cpp:80`
- On‑stream GPU entry (host‑launched kernels)
  - RMA entrypoints: `src/host/comm/rma.cu:11` → kernels `src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh:44`
  - Barrier/sync kernels: `src/host/stream/coll/barrier/barrier.cu:13` → kernels `src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:44`
- Device transfer paths
  - Device API inlines → transfer templates: `src/include/device/nvshmem_defines.h:81`, `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:104`
  - GPU proxy channel (device→host handoff): `src/include/non_abi/device/pt-to-pt/proxy_device.cuh:1`
- Transports
  - Transport interfaces: `src/include/internal/host_transport/transport.h:119`
  - Libfabric implementation: `src/modules/transport/libfabric/libfabric.cpp:740`
- Python
  - High‑level APIs: `nvshmem4py/nvshmem/core/*.py`
  - Cython bindings: `nvshmem4py/nvshmem/bindings/nvshmem.pyx:1397` (hostlib_init_attr), `nvshmem4py/nvshmem/bindings/cynvshmem.pyx:1`

Initialization & Symmetric Heap

- C++ host apps call `nvshmem_init()` which wraps `nvshmemi_init_thread()`:
  - `src/include/host/nvshmem_api.h:52` (nvshmem_init)
  - `src/device/init/init_device.cu:139` (nvshmemi_init_thread)
    - Calls `nvshmemid_hostlib_init_attr(...)` to bootstrap host runtime and register device state callbacks:
      `src/host/init/init.cu:1220`
    - If fully initialized, sets up device‑only state (`_nvshmemi_init_device_only_state`), resolves CUDA device
  - Host library bootstrap resolves environment and bootstrap method (MPI/UID):
    - `src/host/init/init.cu:1302` (nvshmemx_hostlib_init_attr)
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
  - `src/host/comm/putget.cpp:120` (nvshmemi_p2p_rma_registered)
- Direct transport call for single‑word `P` and chunked multi‑send for bulk:
  - `src/include/internal/host/nvshmem_internal.h:168` (nvshmemi_process_multisend_rma)
  - `src/include/internal/host_transport/transport.h:119` (rma signature)
  - Libfabric RMA implementation: `src/modules/transport/libfabric/libfabric.cpp:740`

2) On‑stream host path (GPU executes the transfer)
- Host composes device kernel args and launches a tiny proxy kernel on the user stream:
  - `src/host/comm/rma.cu:11` (nvshmemi_proxy_rma_launcher)
  - `src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh:44` (nvshmemi_proxy_rma_entrypoint[_blocking])
- Those kernels call device transfer templates, which pick between IBGDA (GPU‑NIC) or GPU→Host proxy channel:
  - `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:104` (nvshmemi_transfer_rma<>)
  - GPU proxy channel details: `src/include/non_abi/device/pt-to-pt/proxy_device.cuh:198` (copy_to_channel, transfer_dma, quiet, fences)

RMA & Signals (Representative APIs)
- Typed/size‑generic host puts call the same helper:
  - `src/host/comm/putget.cpp:364` (nvshmem_putmem)
  - `src/host/comm/putget.cpp:375` (nvshmemx_putmem_on_stream)
  - `src/host/comm/putget.cpp:511` (nvshmemx_putmem_signal_on_stream)
- Device API inlines expand to transfer templates and signal ops:
  - `src/include/device/nvshmem_defines.h:81` (nvshmem_TYPENAME_p)
  - `src/include/device/nvshmem_defines.h:107` (nvshmem_TYPENAME_put_signal)

Transports (Host)

- Selection and posting
  - The helper selects `nvshmemi_state->selected_transport_for_rma[pe]` and obtains memory handles for local/remote symmetric memory or registered buffers:
    - `src/include/internal/host/nvshmem_internal.h:130` (nvshmemi_get_local_mem_handle)
    - `src/include/internal/host/nvshmem_internal.h:152` (nvshmemi_get_remote_mem_handle)
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
- User (C++) calls `nvshmemx_putmem_signal_on_stream`:
  - `src/host/comm/putget.cpp:511` → nvshmemi_prepare_and_post_rma `src/host/comm/putget.cpp:232`
  - If peer is directly mapped (same NVSwitch domain), cudaMemcpyAsync + `nvshmemi_signal_op_on_stream` (`src/host/comm/sync.cpp:102`).
  - Else offloads to device proxy kernel: `src/host/comm/rma.cu:11` → `src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh:75` (signal entrypoint)
  - Device template executes put and signal: `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:137` (nvshmemi_transfer_put_signal)
  - Transport work performed by IBGDA or by writing GPU→Host proxy channel entries `src/include/non_abi/device/pt-to-pt/proxy_device.cuh:206`.

2) Device‑initiated `nvshmem_float_put` in a kernel
- Device API inline: `src/include/device/nvshmem_defines.h:98` → `nvshmemi_put<T,SCOPE>` template (included through transfer headers)
- Template selects IBGDA or proxy channel path: `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:104`
  - Proxy writes requests into ring‑buffer (channel) consumed by host proxy thread: `src/include/non_abi/device/pt-to-pt/proxy_device.cuh:206`

3) Python `nvshmem.core.reduce(..., stream=...)`
- Core wrapper locates dtype/op and resolves binding function name:
  - `nvshmem4py/nvshmem/core/collective.py:110` → `collective_on_buffer` → `bindings.<dtype>_<op>_reduce_on_stream`
- Cython binding calls C++ host API inlined in headers, implemented in `src/host/coll/*` and on‑stream kernels in `src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:44`.

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

