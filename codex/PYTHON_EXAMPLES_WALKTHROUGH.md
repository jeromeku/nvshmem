**Python Examples: Annotated Call Paths (nvshmem4py/examples)**

This document traces Python calls down through nvshmem4py core wrappers, Cython bindings, C++ host APIs, and device kernels. Inline code and VSCode‑style file links show the full path from Python to the lowest level used.

Conventions
- VSCode links: path:line
- Snippets trimmed to the call sites and key transitions.

Shared Concepts
- Python nvshmem4py has two tiers:
  - High‑level convenience APIs in `nvshmem4py/nvshmem/core/*.py` that accept Buffers, arrays (CuPy), and tensors (PyTorch) and perform validation.
  - Thin Cython bindings in `nvshmem4py/nvshmem/bindings/*.pyx` that call the C++ host APIs (and device inlines via on‑stream kernels).
- Device‑side APIs for Numba are enabled by injecting NVSHMEM headers into NVRTC via `nvshmem4py/nvshmem/bindings/device/numba/entry_point.h:1`.

Init/Finalize (common prelude)

Python code
```python
# nvshmem4py/examples/on-stream.py:50
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
...
nvshmem.core.finalize()
```
- File: nvshmem4py/examples/on-stream.py:50

Call path
- Core init orchestrates bootstrap selection and calls bindings:
  - File: nvshmem4py/nvshmem/core/init_fini.py:112 (init)
  - File: nvshmem4py/nvshmem/bindings/nvshmem.pyx:1397 (hostlib_init_attr)
  - Hostlib init: File: src/host/init/init.cu:1302 (nvshmemx_hostlib_init_attr) → common init: File: src/host/init/init.cu:994
- Finalize:
  - File: nvshmem4py/nvshmem/core/init_fini.py:214 (finalize)
  - File: nvshmem4py/nvshmem/bindings/nvshmem.pyx:1403 (hostlib_finalize)
  - Host: File: src/host/init/init.cu:1434 (nvshmemx_hostlib_finalize → nvshmemid_hostlib_finalize)

Memory: nvshmem.core.array / buffer

Python code
```python
# nvshmem4py/examples/on-stream.py:59
input = nvshmem.core.array((input_nelems,), dtype="int")
partial_sum = nvshmem.core.array((1,), dtype="int")
full_sum = nvshmem.core.array((1,), dtype="int")
```
- File: nvshmem4py/examples/on-stream.py:59–61

Call path
- CuPy interop allocates an NVSHMEM buffer then wraps with CuPy
  - File: nvshmem4py/nvshmem/core/interop/cupy.py:72 (array) → `nvshmem.core.buffer(size)`
  - File: nvshmem4py/nvshmem/core/memory.py:43 (buffer)
    - Alloc path (MemoryResource): File: nvshmem4py/nvshmem/core/nvshmem_types.py:173 (NvshmemResource.allocate)
    - Cython binding malloc: File: nvshmem4py/nvshmem/bindings/cynvshmem.pyx:26 (nvshmem_malloc)
    - Host malloc: File: src/host/mem/mem_heap.cpp:2288 (nvshmem_malloc)

Example 1: Host‑initiated collective on stream (on-stream.py)

Python code
```python
# nvshmem4py/examples/on-stream.py:64
nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, full_sum, partial_sum, "sum", stream=stream)
```
- File: nvshmem4py/examples/on-stream.py:64

Call path
- High‑level collective resolves dtype/op and dispatches to buffer‑level function
  - File: nvshmem4py/nvshmem/core/collective.py:110 (collective_on_buffer)
  - Resolves binding function name `<dtype>_<op>_reduce_on_stream` and calls it with raw Buffer handles
- Cython binding calls C++ host extended API
  - File: nvshmem4py/nvshmem/bindings/nvshmem.pyx: (e.g., float/int reduce_on_stream cpdefs)
  - Maps to `nvshmemx_*_reduce_on_stream` declared in `src/include/host/nvshmemx_api.h`
- Host on‑stream launcher binds a CUDA kernel for the team and stream
  - File: src/host/stream/coll/barrier/barrier.cu:13 (pattern; coll entrypoints are similar)
  - Device collective kernels: File: src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:104 (rdxn_on_stream_kernel)
  - Device kernels use threadgroup collectives (reduce, reducescatter) under `src/include/non_abi/device/coll/*.cuh`

Example 2: Host‑initiated RMA with signal on stream (rma.py)

Python code
```python
# nvshmem4py/nvshmem/core/rma.py:137
nvshmem.core.put_signal(dst, src, signal_var, signal_val, signal_op, remote_pe, stream)
```
- File: nvshmem4py/nvshmem/core/rma.py:137

Call path
- High‑level `_call_putget` validates and resolves `putmem_signal_on_stream` binding
  - File: nvshmem4py/nvshmem/core/rma.py:76 (helper), 119 (name resolution)
- Cython binding (untyped putmem_signal_on_stream):
  - File: nvshmem4py/nvshmem/bindings/nvshmem.pxd:246 (cpdef), `nvshmem4py/nvshmem/bindings/nvshmem.pyx`
  - C++ host API: `src/host/comm/putget.cpp:511` (nvshmemx_putmem_signal_on_stream)
- Host path
  - Mapped peer → cudaMemcpyAsync + `nvshmemi_signal_op_on_stream`: File: `src/host/comm/sync.cpp:102`
  - Else launches device proxy kernel via `nvshmemi_proxy_rma_signal_entrypoint`:
    - File: src/host/comm/rma.cu:11 → `src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh:75`
  - Device transfer template with signal: `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:137`

Example 3: Device‑initiated ring reduce in Numba (device_custom_kernel.py)

Python code
```python
# nvshmem4py/examples/device_custom_kernel.py:25
@cuda.jit(lto=True)
def ring_reduce(dst, src, nreduce, signal, chunk_size):
    mype = my_pe()
    npes = n_pes()
    ...
    if thread_id == 0:
        signal_wait_until(signal_block, CMP_GE, chunk + 1)
    ...
    if thread_id == 0:
        put_signal_nbi(dst_block, src_data, chunk_elems, signal_block, 1, SIGNAL_ADD, peer)
```
- File: nvshmem4py/examples/device_custom_kernel.py:25–52

Device call path
- Numba device bindings expose in‑kernel functions by including NVSHMEM headers at NVRTC compile time:
  - File: nvshmem4py/nvshmem/bindings/device/numba/entry_point.h:1 (`#include <nvshmem.h>`, `<nvshmemx.h>`)
- Device API inlines route to internal device transfer templates:
  - `nvshmem_signal_wait_until`: File: src/include/device/nvshmem_defines.h:583
  - `nvshmem_put_signal_nbi`: put + signal device template: File: src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in:137
  - Templates select IBGDA or GPU→Host proxy channel ops based on `nvshmemi_device_state_d` flags.

Example 4: Torch/Triton interop (torch_triton_interop.py)

Python code (abbrev.)
```python
# nvshmem4py/examples/torch_triton_interop.py:49
uniqueid = nvshmem.core.get_unique_id()
...
nvshmem.core.init(device=dev, uid=broadcast_objects[0], rank=rank_id,
                  nranks=num_ranks, initializer_method="uid")
...
tensor1 = nvshmem.core.tensor((n_elements,), dtype=torch.float32)
nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, tensor_out, tensor_sum, "sum", stream=stream)
```
- Files: nvshmem4py/examples/torch_triton_interop.py:49, 61, 103

Call path
- UID bootstrap in Python
  - `get_unique_id`: File: nvshmem4py/nvshmem/core/init_fini.py:72 → `nvshmem4py/nvshmem/bindings/nvshmem.pyx` (`get_uniqueid`) → C++ host `nvshmemx_get_uniqueid` (decl.
    in `src/include/host/nvshmemx_api.h`)
  - `init(..., initializer_method="uid")`: File: nvshmem4py/nvshmem/core/init_fini.py:144 → `set_attr_uniqueid_args(...)` + `hostlib_init_attr`
- Tensor allocation path uses NVSHMEM buffer interop for PyTorch similar to CuPy arrays (via DLPack or torch‑specific wrappers under `nvshmem/core/interop/torch.py`).
- Collectives from PyTorch tensors map to the same binding functions as CuPy arrays via unified buffer accessors.

Example 5: Triton comm kernels (triton_comm_kernels.py)

Python code (abbrev.)
```python
# nvshmem4py/examples/triton_comm_kernels.py:139
remote_mc_tensor = nvshmem.get_multicast_tensor(nvshmem.Teams.TEAM_WORLD, tensor)
...
nvshmem.barrier(nvshmem.Teams.TEAM_WORLD, stream=stream)
```
- Files: nvshmem4py/examples/triton_comm_kernels.py:139, 151

Call path
- Multicast pointer acquisition:
  - `get_multicast_tensor` → `get_multicast_buffer` → `bindings.mc_ptr`:
    - File: nvshmem4py/nvshmem/core/memory.py:319 (get_mc_buffer)
    - File: nvshmem4py/nvshmem/bindings/cynvshmem.pyx:43 (nvshmemx_mc_ptr)
    - Host API: `nvshmemx_mc_ptr` decl. in `src/include/host/nvshmem_api.h:93`
- Barrier on stream:
  - High‑level wrapper: `nvshmem.core.barrier(..., stream=...)` (collective.py)
  - Binding: `nvshmemx_barrier_on_stream` (cpdef)
  - Host: File: src/host/coll/barrier/barrier_on_stream.cpp:80 → `nvshmemi_call_barrier_on_stream_kernel`
  - Device kernel: File: src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh:44

Additional Notes
- Stream interop: Python stream objects implement `__cuda_stream__` so on‑stream bindings receive a raw CUDA stream handle: `nvshmem4py/nvshmem/core/nvshmem_types.py:147` (NvshmemStream)
- External buffers: `register_external_buffer` allows operating on non‑NVSHMEM allocations by registering them symmetrically (collective put/get and peer pointer operations require symmetric or registered buffers):
  - File: nvshmem4py/nvshmem/core/memory.py:187 (register_external_buffer)
  - Bindings: `buffer_register_symmetric` in `nvshmem4py/nvshmem/bindings/_internal/*.pyx`

