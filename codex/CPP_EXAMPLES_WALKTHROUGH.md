**C++ Examples: Annotated Call Paths**

This document traces the call stacks for representative examples under `examples/`, expanding API calls through host/device layers to transports and kernels. Each section starts at the user‑facing API and fully unfurls the path, with inline code and file links.

Conventions
- VSCode links: path:line
- Snippets are trimmed to focus on the callsite and immediately relevant code.

Legend
- Host = CPU runtime path (off‑stream)
- On‑stream = host launches small GPU kernel to perform transfer/sync on the GPU
- Device = GPU‑initiated inline API path

Hello World (hello.cpp)

User code
```cpp
// examples/hello.cpp:22
nvshmem_init();
...
void *ptr = nvshmem_malloc(1);
...
nvshmem_finalize();
```
- File: examples/hello.cpp:22
- File: examples/hello.cpp:26
- File: examples/hello.cpp:31

Call path
- `nvshmem_init()`
  - File: src/include/host/nvshmem_api.h:52
  - Wraps `nvshmemi_init_thread()` in device lib:
    - File: src/device/init/init_device.cu:139
    - Calls `nvshmemid_hostlib_init_attr(...)` to bootstrap host runtime:
      - File: src/host/init/init.cu:1220
    - Common init builds transports, symmetric heap, teams:
      - File: src/host/init/init.cu:994
- `nvshmem_malloc` allocates from the symmetric heap and barriers:
  - File: src/host/mem/mem_heap.cpp:2288
  - Checks initialized state: File: src/host/init/init.cu:1195
- `nvshmem_finalize` → `nvshmemi_finalize` (host lib finalize and device state unregister):
  - File: src/device/init/init_device.cu:171

Block‑Scoped Put (put-block.cu)

User kernel
```cpp
// examples/put-block.cu:48
nvshmemx_float_put_block(recv_data + block_offset,
                         send_data + block_offset,
                         min(blockDim.x, num_elems - block_offset),
                         peer);
```
- File: examples/put-block.cu:48

Device API → transfer templates
- `nvshmemx_*_put_block` is declared in host extended header (device decls):
  - File: src/include/host/nvshmemx_api.h:280
- Device inline wrappers route to internal transfer templates (block scope):
  - File: src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:104
    - If IBGDA available, issue device‑driven NIC work; else write requests into GPU→Host proxy channel.
- GPU→Host proxy channel (when not IBGDA):
  - Ring buffer write and flow control:
    - File: src/include/non_abi/device/pt-to-pt/proxy_device.cuh:206

Host init + memory setup (main)
```cpp
// examples/put-block.cu:70
nvshmem_init();
...
cudaSetDevice(mype_node);
send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
```
- Files: examples/put-block.cu:70, examples/put-block.cu:80–82
- `nvshmem_init`/`nvshmem_malloc` paths as above.

On‑Stream Collectives (on-stream.cu)

User code
```cpp
// examples/on-stream.cu:83
accumulate<<<1, input_nelems, 0, stream>>>(input, partial_sum);
// on‑stream reduction (host API)
nvshmemx_int_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, full_sum, partial_sum, to_all_nelems, stream);
correct_accumulate<<<1, input_nelems, 0, stream>>>(input, partial_sum, full_sum);
```
- File: examples/on-stream.cu:83
- File: examples/on-stream.cu:84

Call path (reduce on stream)
- `nvshmemx_int_sum_reduce_on_stream(...)` declaration in extended host header
  - File: src/include/host/nvshmemx_api.h: (reduce family, on stream)
- Host launches a small on‑stream kernel to perform collective on device team/psync structures:
  - Entry glue chooses appropriate threadgroup kernel:
    - File: src/host/stream/coll/barrier/barrier.cu:13 (pattern used for coll on‑stream entrypoints)
  - Collective kernels (reduce/scatter/broadcast) are defined as `__global__` templates:
    - File: src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:104 (rdxn_on_stream_kernel)
  - Device collective invokes transfer templates (P2P/IBGDA or proxy) and synchronization across threadgroups:
    - File: src/include/non_abi/device/coll/reduce.cuh
    - File: src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:104

GPU‑Side Ring Reduce with Signal (ring-reduce.cu)

Kernel (device‑initiated signal + put)
```cpp
// examples/ring-reduce.cu:114
if (thread_id == 0) nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, chunk + 1);
...
// send data and atomically signal peer
nvshmem_int_put_signal_nbi(dst, (mype == 0) ? src : dst,
                           chunk_elems, signal, 1, NVSHMEM_SIGNAL_ADD, peer);
```
- Files: examples/ring-reduce.cu:114, examples/ring-reduce.cu:123

Device call path
- Device signal wait templates/inlines in public headers:
  - File: src/include/host/nvshmem_api.h:335+ (device decls)
  - File: src/include/device/nvshmem_defines.h:578 (signal fetch/wait)
- Device `put_signal_nbi` → transfer template with signal
  - File: src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:137 (nvshmemi_transfer_put_signal)
  - If IBGDA unavailable, proxy channel request `nvshmemi_proxy_put_signal_nbi` then quiet when needed.

MPI‑Based Bootstrap (mpi-based-init.cu)

User code
```cpp
// examples/mpi-based-init.cu:60
nvshmem_init();          // OR nvshmemi_init_mpi(...) then nvshmem_init()
...
__global__ void simple_shift(int *target, int mype, int npes) {
    int peer = (mype + 1) % npes;
    nvshmem_int_p(target, mype, peer);
}
```
- Files: examples/mpi-based-init.cu:60, examples/mpi-based-init.cu:38–41

Call path
- `nvshmem_init` uses MPI bootstrap when configured via environment or helper (`nvshmemi_init_mpi`) and proceeds as in Hello example:
  - File: src/host/init/init.cu:1220 (nvshmemid_hostlib_init_attr)
- Device `nvshmem_int_p` → inlines:
  - File: src/include/device/nvshmem_defines.h:81
  - Then device transfer: `transfer_device.cuh.in:67` (nvshmemi_transfer_rma_p)

UID‑Based Bootstrap (uid-based-init.cu)

User code (abbrev.)
```cpp
// examples/uid-based-init.cu:46
nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
...
nvshmemx_get_uniqueid(&id);
MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
```
- File: examples/uid-based-init.cu:46–60

Call path
- Hostlib init with UID attributes mirrors Python UID bootstrap:
  - File: src/host/init/init.cu:1220 (nvshmemid_hostlib_init_attr)
  - Populates device state and proceeds into `nvshmemi_common_init` `src/host/init/init.cu:994`

On‑Stream Barrier (barrier.cpp path)

User code (pattern)
```cpp
// examples/on-stream.cu:84 (reduce uses the same on‑stream mechanism)
nvshmemx_int_sum_reduce_on_stream(..., stream);
```
- Host on‑stream barrier entry:
  - File: src/host/coll/barrier/barrier.cpp:20 (nvshmem_barrier) → `nvshmemi_call_barrier_on_stream_kernel`
  - File: src/host/stream/coll/barrier/barrier.cu:13 (launches threadgroup kernel)
- Device kernel (dissemination, consistency at target as needed):
  - File: src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:44 (barrier_on_stream_kernel_threadgroup)
  - Algorithms: `src/include/non_abi/device/coll/barrier.cuh:196`

Notes
- Many examples have conditional MPI helpers; bootstrap helper headers are provided under `examples/bootstrap_helper.h`.
- The host runtime paths converge on `nvshmemi_prepare_and_post_rma` for put/get and per‑collective on‑stream launchers for collectives.

