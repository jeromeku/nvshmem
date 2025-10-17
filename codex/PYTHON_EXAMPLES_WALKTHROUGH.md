**Python Examples: Annotated Call Paths (nvshmem4py/examples)**

This document traces Python calls through nvshmem4py → Cython bindings → C++ host APIs → device kernels/transports, with commentary at each hop and clickable links.

Conventions
- Links use file anchors with line numbers, for example: [collective.py:15](nvshmem4py/nvshmem/core/collective.py#L15)
- Snippets are trimmed to the relevant call sites.

Shared Concepts
- High‑level APIs live in nvshmem4py’s `core` package and accept CUDA Buffers or framework arrays/tensors.
- Bindings (Cython) dispatch to C++ NVSHMEM host/device APIs.
- Numba device APIs work by injecting NVSHMEM headers into NVRTC includes: [entry_point.h:1](nvshmem4py/nvshmem/bindings/device/numba/entry_point.h#L1)

Init/Finalize (common prelude)

Python code
```python
# nvshmem4py/examples/on-stream.py:50
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
...
nvshmem.core.finalize()
```

Call path — what happens
- nvshmem4py orchestrates bootstrap and calls bindings:
  - [init_fini.py:112](nvshmem4py/nvshmem/core/init_fini.py#L112) (init)
  - [nvshmem.pyx:1397](nvshmem4py/nvshmem/bindings/nvshmem.pyx#L1397) (hostlib_init_attr)
  - NVSHMEM hostlib init and common init: [init.cu:1302](src/host/init/init.cu#L1302) → [init.cu:994](src/host/init/init.cu#L994)
- Finalize performs buffer cleanup and hostlib finalize:
  - [init_fini.py:214](nvshmem4py/nvshmem/core/init_fini.py#L214) (finalize)
  - [nvshmem.pyx:1403](nvshmem4py/nvshmem/bindings/nvshmem.pyx#L1403) (hostlib_finalize)

Memory: nvshmem.core.array / buffer

Python code
```python
# nvshmem4py/examples/on-stream.py:59–61
arr = nvshmem.core.array((n,), dtype="int")
```

Call path — how memory is created
- CuPy interop allocates an NVSHMEM Buffer and wraps it as a CuPy ndarray:
  - [cupy.py:72](nvshmem4py/nvshmem/core/interop/cupy.py#L72) (array) → nvshmem.core.buffer(size)
  - [memory.py:43](nvshmem4py/nvshmem/core/memory.py#L43) (buffer)
    - NvshmemResource allocates via bindings: [nvshmem_types.py:173](nvshmem4py/nvshmem/core/nvshmem_types.py#L173)
    - Cython binding malloc: [cynvshmem.pyx:26](nvshmem4py/nvshmem/bindings/cynvshmem.pyx#L26)
    - Host malloc: [mem_heap.cpp:2288](src/host/mem/mem_heap.cpp#L2288)

Example 1: Host‑initiated collective on stream (on-stream.py)

Python code
```python
# nvshmem4py/examples/on-stream.py:64
nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, full_sum, partial_sum, "sum", stream=stream)
```

Call path — layer by layer
- High‑level resolves dtype/op and dispatches to Buffer‑level routine:
  - [collective.py:110](nvshmem4py/nvshmem/core/collective.py#L110) (collective_on_buffer)
- Binding function `<dtype>_<op>_reduce_on_stream` is invoked:
  - cpdefs in [nvshmem.pyx](nvshmem4py/nvshmem/bindings/nvshmem.pyx)
- Host on‑stream launcher enqueues a device kernel:
  - [barrier.cu:13](src/host/stream/coll/barrier/barrier.cu#L13) (pattern used for coll on‑stream)
- Device collective kernel performs the reduction:
  - [nvshmemi_h_to_d_coll_defs.cuh:104](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L104)

Example 2: Host‑initiated RMA with signal on stream (rma.py)

Python code
```python
# nvshmem4py/nvshmem/core/rma.py:137
nvshmem.core.put_signal(dst, src, sig, val, op, pe, stream)
```

Call path — layer by layer
- High‑level `_call_putget` validates inputs and resolves binding name:
  - [rma.py:76](nvshmem4py/nvshmem/core/rma.py#L76), [rma.py:118](nvshmem4py/nvshmem/core/rma.py#L118)–[rma.py:130](nvshmem4py/nvshmem/core/rma.py#L130)
- Cython binding (untyped putmem_signal_on_stream):
  - [nvshmem.pxd:246](nvshmem4py/nvshmem/bindings/nvshmem.pxd#L246), [nvshmem.pyx](nvshmem4py/nvshmem/bindings/nvshmem.pyx)
- Host path for mapped vs. remote peer:
  - Mapped peer → cudaMemcpyAsync + signal op: [sync.cpp:102](src/host/comm/sync.cpp#L102)
  - Else launch proxy kernel → device entrypoint: [rma.cu:11](src/host/comm/rma.cu#L11) → [nvshmemi_h_to_d_rma_defs.cuh:75](src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh#L75)
- Device put+signal template executes on GPU:
  - [transfer_device.cuh.in:137](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L137)

Example 3: Device‑initiated ring reduce in Numba (device_custom_kernel.py)

Python code
```python
# nvshmem4py/examples/device_custom_kernel.py:25–52
@cuda.jit(lto=True)
def ring_reduce(...):
    if thread_id == 0:
        signal_wait_until(signal_block, CMP_GE, chunk + 1)
    if thread_id == 0:
        put_signal_nbi(dst_block, src_data, chunk_elems, signal_block, 1, SIGNAL_ADD, peer)
```

Call path — device focus
- Numba device bindings enable `nvshmem_*` calls via includes: [entry_point.h:1](nvshmem4py/nvshmem/bindings/device/numba/entry_point.h#L1)
- Device wait/signal and put+signal resolve to inlines/templates:
  - Wait/signal inlines: [nvshmem_defines.h:583](src/include/device/nvshmem_defines.h#L583)
  - Put+signal template: [transfer_device.cuh.in:137](src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in#L137)
- IBGDA vs. proxy channel is chosen by device state flags.

Example 4: Torch/Triton interop (torch_triton_interop.py)

Python code (UID bootstrap + reduce on a stream)
```python
# nvshmem4py/examples/torch_triton_interop.py:49,61,103
uniqueid = nvshmem.core.get_unique_id()
nvshmem.core.init(device=dev, uid=uniqueid, rank=rank_id, nranks=num_ranks, initializer_method="uid")
nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, tensor_out, tensor_sum, "sum", stream=stream)
```

Call path — bootstrap and reduce
- UID retrieval via bindings: [init_fini.py:72](nvshmem4py/nvshmem/core/init_fini.py#L72) → [nvshmem.pyx](nvshmem4py/nvshmem/bindings/nvshmem.pyx)
- UID init through hostlib: [init_fini.py:144](nvshmem4py/nvshmem/core/init_fini.py#L144) → [nvshmem.pyx](nvshmem4py/nvshmem/bindings/nvshmem.pyx)
- Reduce follows Example 1’s on‑stream path.

Example 5: Triton comm kernels (triton_comm_kernels.py)

Python code (multicast view + barrier)
```python
# nvshmem4py/examples/triton_comm_kernels.py:139,151
mc = nvshmem.get_multicast_tensor(nvshmem.Teams.TEAM_WORLD, tensor)
nvshmem.barrier(nvshmem.Teams.TEAM_WORLD, stream=stream)
```

Call path — multicast and barrier
- Multicast pointer acquisition:
  - Core: [memory.py:319](nvshmem4py/nvshmem/core/memory.py#L319) (get_mc_buffer) → bindings mc_ptr
  - Binding: [cynvshmem.pyx:43](nvshmem4py/nvshmem/bindings/cynvshmem.pyx#L43)
- On‑stream barrier:
  - Binding: [nvshmem.pxd](nvshmem4py/nvshmem/bindings/nvshmem.pxd)
  - Host enqueue: [barrier_on_stream.cpp:80](src/host/coll/barrier/barrier_on_stream.cpp#L80) → device kernel launcher
  - Device kernel: [nvshmemi_h_to_d_coll_defs.cuh:44](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L44)

Example 6: All‑to‑All and Fcollect on Buffers (collective.py)

Python code (buffer path)
```python
# nvshmem4py/nvshmem/core/collective.py:112
collective_on_buffer(coll, team, dest_buf, src_buf, dtype, op, root, stream)
```

Call path — name resolution & launch
- Resolves function name `<dtype>_<op?>_<coll>_on_stream` and calls binding:
  - [collective.py:135](nvshmem4py/nvshmem/core/collective.py#L135)–[collective.py:140](nvshmem4py/nvshmem/core/collective.py#L140)
- Bindings for alltoall/fcollect map to host APIs defined in `nvshmemx_*`:
  - [nvshmem.pxd](nvshmem4py/nvshmem/bindings/nvshmem.pxd)
- On‑stream kernels:
  - All‑to‑all: [nvshmemi_h_to_d_coll_defs.cuh:16](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L16)
  - Fcollect: [nvshmemi_h_to_d_coll_defs.cuh:124](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L124)

Example 7: External Tensor Registration + Reduce (torch_mempool.py)

Python code
```python
# nvshmem4py/examples/torch_mempool.py:141,145
registered = nvshmem.core.register_external_tensor(tensor)
nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, registered, registered, op="sum", stream=dev.create_stream())
```

Call path — register then reuse standard paths
- Registration delegates to Buffer registration:
  - [memory.py:187](nvshmem4py/nvshmem/core/memory.py#L187) (register_external_buffer)
- Reduce follows Example 1 once the buffer is registered.

Example 8: Peer Array View and Multicast Array (CuPy)

Python code
```python
peer = nvshmem.core.get_peer_array(array, peer)
mc   = nvshmem.core.get_multicast_array(nvshmem.core.Teams.TEAM_WORLD, array)
```

Call path — pointer aliasing
- Peer pointer (nvshmem_ptr):
  - [cupy.py:98](nvshmem4py/nvshmem/core/interop/cupy.py#L98) (get_peer_array)
  - [memory.py:119](nvshmem4py/nvshmem/core/memory.py#L119) (get_peer_buffer)
- Multicast pointer (nvshmemx_mc_ptr):
  - [cupy.py:126](nvshmem4py/nvshmem/core/interop/cupy.py#L126)
  - [memory.py:287](nvshmem4py/nvshmem/core/memory.py#L287)

Example 9: Simple P2P Kernel (simple_p2p_kernel.py)

Python code (device write + barrier)
```python
# nvshmem4py/examples/simple_p2p_kernel.py
@cuda.jit
def simple_shift(arr, dst_pe):
    arr[0] = dst_pe
...
nvshmem.core.barrier(nvshmem.core.Teams.TEAM_NODE, stream)
```

Call path — minimal device action + host barrier
- Device kernel writes a value; no nvshmem device call here (demonstrates use of symmetric arrays in kernels).
- Barrier on stream flows through bindings to host on‑stream barrier:
  - Wrapper in core.collective: [collective.py](nvshmem4py/nvshmem/core/collective.py)
  - Host enqueue: [barrier_on_stream.cpp:80](src/host/coll/barrier/barrier_on_stream.cpp#L80)
  - Device kernel: [nvshmemi_h_to_d_coll_defs.cuh:44](src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh#L44)
  - nvshmem4py/nvshmem/core/interop/cupy.py:L126 (get_multicast_array) → `get_multicast_buffer`
  - nvshmem4py/nvshmem/core/memory.py:L287 (get_mc_buffer) → host `nvshmemx_mc_ptr`
