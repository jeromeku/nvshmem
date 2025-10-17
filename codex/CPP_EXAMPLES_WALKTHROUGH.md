**C++ Examples: Annotated Call Paths**

This guided tour follows representative examples under `examples/`, and for each, walks the data/control flow through user code → NVSHMEM APIs → host/runtime layers → device kernels and transports. Links are clickable in VSCode.

Conventions
- Links use file anchors with line numbers, for example: [on-stream.cu:84](examples/on-stream.cu#L84)
- Snippets are trimmed to the relevant call sites, with commentary explaining what happens at each hop.

Legend
- Host = CPU runtime path (off‑stream)
- On‑stream = host launches a GPU kernel to do work/sync on device
- Device = GPU‑initiated inline NVSHMEM APIs

Hello World (hello.cpp)

User code (sets up NVSHMEM, allocates symmetric heap, prints PE info)
```cpp
// examples/hello.cpp:22
nvshmem_init();
...
void *ptr = nvshmem_malloc(1);
...
nvshmem_finalize();
```
- [hello.cpp:22](examples/hello.cpp#L22)
- [hello.cpp:26](examples/hello.cpp#L26)
- [hello.cpp:31](examples/hello.cpp#L31)

Call path — what each layer does
- Init shim → internal thread init → hostlib bootstrap → common init: 
  - [nvshmem_api.h:52](src/include/host/nvshmem_api.h#L52) calls into the device library entry
  - [init_device.cu:139](src/device/init/init_device.cu#L139) arranges host‑side init from device context
  - [init.cu:1220](src/host/init/init.cu#L1220) registers the hostlib and bootstraps (MPI/UID, env)
  - [init.cu:994](src/host/init/init.cu#L994) builds the transport map, heap, teams, and proxy
- Symmetric allocation + symmetry: 
  - [mem_heap.cpp:2288](src/host/mem/mem_heap.cpp#L2288) allocates from the symmetric heap and enters a barrier to enforce collective symmetry
- Finalization: 
  - [init_device.cu:171](src/device/init/init_device.cu#L171) unregisters device state and finalizes the hostlib

Block‑Scoped Put (put-block.cu)

User kernel (all threads in a block cooperatively issue a put for bandwidth)
```cpp
// examples/put-block.cu:48
nvshmemx_float_put_block(recv_data + block_offset,
                         send_data + block_offset,
                         min(blockDim.x, num_elems - block_offset),
                         peer);
```
- [put-block.cu:48](examples/put-block.cu#L48)

How it flows
- Device entrypoints for block/warp/thread scopes: [nvshmemx_api.h:280](src/include/host/nvshmemx_api.h#L280)
- Device transfer selection:
  - [transfer_device.cuh.in:104](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L104) chooses IBGDA (GPU→NIC) when enabled, else encodes a request for the host proxy
- Proxy channel mechanics:
  - [proxy_device.cuh:206](src/include/non_abi/device/pt-to-pt/proxy_device.cuh#L206) pushes DMA requests into a GPU‑resident ring buffer polled by the proxy thread

Host init + memory setup (example main)
```cpp
// examples/put-block.cu:70,80–82
nvshmem_init();
cudaSetDevice(mype_node);
send_data = (float *)nvshmem_malloc(...);
recv_data = (float *)nvshmem_malloc(...);
```
- The same init/malloc paths from Hello apply here.

On‑Stream Collectives (on-stream.cu)

User code (host reduction on a user stream between two compute kernels)
```cpp
// examples/on-stream.cu:83–86
accumulate<<<... , stream>>>(...);
nvshmemx_int_sum_reduce_on_stream(..., stream);
correct_accumulate<<<... , stream>>>(...);
```
- [on-stream.cu:83](examples/on-stream.cu#L83)
- [on-stream.cu:84](examples/on-stream.cu#L84)

How it flows
- Host API (extended) exposes on‑stream collectives: [nvshmemx_api.h:36](src/include/host/nvshmemx_api.h#L36)
- The launcher binds a small CUDA kernel on the provided stream:
  - [barrier.cu:13](src/host/stream/coll/barrier/barrier.cu#L13) shows the general on‑stream launch pattern
- Device collective kernels implement the algorithm:
  - [nvshmemi_h_to_d_coll_defs.cuh:104](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L104) (reduce), others for fcollect/alltoall
- Data movement during collectives uses device transfer templates:
  - [transfer_device.cuh.in:104](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L104)

GPU‑Side Ring Reduce with Signal (ring-reduce.cu)

Kernel (device‑initiated signal + put; reduce chunks locally, then forward)
```cpp
// examples/ring-reduce.cu:114,123
if (thread_id == 0) nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, chunk + 1);
nvshmem_int_put_signal_nbi(dst, (mype == 0) ? src : dst, chunk_elems, signal, 1, NVSHMEM_SIGNAL_ADD, peer);
```
- [ring-reduce.cu:114](examples/ring-reduce.cu#L114), [ring-reduce.cu:123](examples/ring-reduce.cu#L123)

How it flows
- Device signal/wait API: [nvshmem_api.h:335](src/include/host/nvshmem_api.h#L335) (decls), [nvshmem_defines.h:578](src/include/device/nvshmem_defines.h#L578) (inlines)
- Device put+signal template: [transfer_device.cuh.in:137](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L137)
- If IBGDA is disabled, the kernel enqueues a proxy request: [proxy_device.cuh:296](src/include/non_abi/device/pt-to-pt/proxy_device.cuh#L296)

Collective Launch (collective-launch.cu)

User code (launch one kernel across PEs as a collective)
```cpp
// examples/collective-launch.cu:85
nvshmemx_collective_launch((const void *)reduce_ring, dimGrid, dimBlock, args, 0, 0);
```
- Extended API: [nvshmemx_collective_launch_apis.h:19](src/include/device/nvshmemx_collective_launch_apis.h#L19)

How it flows
- Host wires the kernel and the team configuration into a coordinated launch; device code uses team state/dups in collectives.
- In‑kernel communication uses device P puts and device collectives:
  - [collective-launch.cu:45](examples/collective-launch.cu#L45), [collective-launch.cu:46](examples/collective-launch.cu#L46)
  - [nvshmem_defines.h:81](src/include/device/nvshmem_defines.h#L81) for `p`

Thread‑Group Collectives (thread-group.cu)

User kernel (compare threadgroup‑wide vs single‑thread fcollect)
```cpp
// examples/thread-group.cu:40–46
nvshmemx_int_fcollect_block(NVSHMEM_TEAM_WORLD, sum, partial_sum, nelems);
// vs
if (index == 0) nvshmem_int_fcollect(NVSHMEM_TEAM_WORLD, sum, partial_sum, nelems);
```
- Entrypoints: [nvshmemx_api.h:280](src/include/host/nvshmemx_api.h#L280)
- Algorithms: [fcollect.cuh](src/include/non_abi/device/coll/fcollect.cuh)
- Threadgroup helpers: [nvshmemi_common_device_defines.cuh:33](src/include/non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh#L33)

User Buffer Registration (user-buffer.cu)

Key calls (register device VMM/EGM allocation, then perform on‑stream collectives)
```cpp
// examples/user-buffer.cu:124,142–143
auto *mmaped = (void *)nvshmemx_buffer_register_symmetric(buffer, size, 0);
nvshmemx_barrier_on_stream(team, stream);
nvshmemx_float_sum_reduce_on_stream(team, dest, source, nelems, stream);
```
- Extended API: [nvshmemx_api.h](src/include/host/nvshmemx_api.h)
- Transport mem‑handle lookups for registered buffers: [nvshmem_internal.h:130](src/include/internal/host/nvshmem_internal.h#L130)
- Collective path is identical to the on‑stream section above once registered.

MPI‑Based Bootstrap (mpi-based-init.cu)

User code (optional MPI helper + device `p`)
```cpp
// examples/mpi-based-init.cu:38,60
__global__ void simple_shift(int *target, int mype, int npes) { nvshmem_int_p(target, mype, peer); }
nvshmem_init(); // or helper + init
```
- Hostlib init on MPI bootstrap: [init.cu:1220](src/host/init/init.cu#L1220)
- Device `p` inlines: [nvshmem_defines.h:81](src/include/device/nvshmem_defines.h#L81)
- Device transfer path: [transfer_device.cuh.in:67](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L67)

UID‑Based Bootstrap (uid-based-init.cu)

User code (UID generation/broadcast + init with attributes)
```cpp
// examples/uid-based-init.cu:46–60
nvshmemx_get_uniqueid(&id);
MPI_Bcast(&id, sizeof(...), ...);
nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
```
- Hostlib init and common init same as above: [init.cu:1220](src/host/init/init.cu#L1220), [init.cu:994](src/host/init/init.cu#L994)

On‑Stream Barrier (host entry + device kernel)
- Host enqueues device barrier/sync kernel on stream:
  - [barrier.cpp:20](src/host/coll/barrier/barrier.cpp#L20) → `nvshmemi_call_barrier_on_stream_kernel`
  - [barrier.cu:13](src/host/stream/coll/barrier/barrier.cu#L13) (launch)
- Device barrier kernel and algorithm:
  - [nvshmemi_h_to_d_coll_defs.cuh:44](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L44)
  - Algorithm: [barrier.cuh:196](src/include/non_abi/device/coll/barrier.cuh#L196)

Ring Broadcast (ring-bcast.cu)

Overview (device pipeline)
- Uses `signal_wait_until` + `put_signal_nbi` to implement a ring broadcast with chunking.
- Device barriers ensure ordering between phases.
- See device wait/signal: [nvshmem_defines.h:578](src/include/device/nvshmem_defines.h#L578) and transfer with signal: [transfer_device.cuh.in:137](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L137)

Dev Guide Ring (dev-guide-ring.cu)

Overview (device put ring)
- Minimal ring that uses `nvshmem_int_p` to forward values among PEs.
- Device `p` inline: [nvshmem_defines.h:81](src/include/device/nvshmem_defines.h#L81); transfer path: [transfer_device.cuh.in:67](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L67)

Call path
- Device threadgroup fcollect decls:
  - src/include/host/nvshmemx_api.h:L280 (threadgroup put/putmem, fcollect symbols near coll device headers)
- Algorithms and threadgroup sync:
  - src/include/non_abi/device/coll/fcollect.cuh (team/dup blocks)
  - Threadgroup helpers: src/include/non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh:L33

User Buffer Registration (user-buffer.cu)

Key calls
```cpp
// examples/user-buffer.cu:L124
mmaped_buffer = (void *)nvshmemx_buffer_register_symmetric(buffer, size, 0);
...
// collectives on registered external buffer
nvshmemx_barrier_on_stream(team, stream);                                  // L142
nvshmemx_float_sum_reduce_on_stream(team, dest, source, nelems, stream);   // L143
```

Call path
- Register external (VMM/EGM) buffer symmetrically:
  - Extended host API decls: src/include/host/nvshmemx_api.h (buffer_register_symmetric)
  - Host registers and maps buffer per PE so transport memory handles can be retrieved:
    - src/include/internal/host/nvshmem_internal.h:L130 (get_local/remote mem handle)
- On‑stream collectives on registered buffers follow the same path as normal symmetric allocations (see on‑stream collectives section above).


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
- examples/mpi-based-init.cu:L60, examples/mpi-based-init.cu:L38

Call path
- `nvshmem_init` uses MPI bootstrap when configured via environment or helper (`nvshmemi_init_mpi`) and proceeds as in Hello example:
  - src/host/init/init.cu:L1220 (nvshmemid_hostlib_init_attr)
- Device `nvshmem_int_p` → inlines:
  - src/include/device/nvshmem_defines.h:L81
  - Device transfer: src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:L67

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
- examples/uid-based-init.cu:L46

Call path
- Hostlib init with UID attributes mirrors Python UID bootstrap:
  - src/host/init/init.cu:L1220 (nvshmemid_hostlib_init_attr)
  - Populates device state and proceeds into nvshmemi_common_init src/host/init/init.cu:L994

On‑Stream Barrier (barrier.cpp path)

User code (pattern)
```cpp
// examples/on-stream.cu:84 (reduce uses the same on‑stream mechanism)
nvshmemx_int_sum_reduce_on_stream(..., stream);
```
- Host on‑stream barrier entry:
- src/host/coll/barrier/barrier.cpp:L20 (nvshmem_barrier) → nvshmemi_call_barrier_on_stream_kernel
- src/host/stream/coll/barrier/barrier.cu:L13 (launches threadgroup kernel)
- Device kernel (dissemination, consistency at target as needed):
  - src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:L44 (barrier_on_stream_kernel_threadgroup)
  - Algorithms: src/include/non_abi/device/coll/barrier.cuh:L196

Notes
- Many examples have conditional MPI helpers; bootstrap helper headers are provided under `examples/bootstrap_helper.h`.
- The host runtime paths converge on `nvshmemi_prepare_and_post_rma` for put/get and per‑collective on‑stream launchers for collectives.
