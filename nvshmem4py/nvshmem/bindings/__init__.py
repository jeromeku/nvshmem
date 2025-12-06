# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information


from .nvshmem import *

import ctypes as _ctypes

from cuda.pathfinder import load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib


_nvshmem_host_lib = None


def _get_nvshmem_host_lib() -> _ctypes.CDLL:
    """Lazily load libnvshmem_host and cache the CDLL handle."""
    global _nvshmem_host_lib
    if _nvshmem_host_lib is not None:
        return _nvshmem_host_lib

    loaded = _load_nvidia_dynamic_lib("nvshmem_host")
    lib_path = loaded.abs_path or "libnvshmem_host.so.3"
    lib = _ctypes.CDLL(lib_path)

    lib.nvshmemx_buffer_register_symmetric.restype = _ctypes.c_void_p
    lib.nvshmemx_buffer_register_symmetric.argtypes = [
        _ctypes.c_void_p,
        _ctypes.c_size_t,
        _ctypes.c_int,
    ]
    lib.nvshmemx_buffer_unregister_symmetric.restype = _ctypes.c_int
    lib.nvshmemx_buffer_unregister_symmetric.argtypes = [
        _ctypes.c_void_p,
        _ctypes.c_size_t,
    ]

    _nvshmem_host_lib = lib
    return lib


def buffer_register_symmetric(ptr: int, size: int, flags: int) -> int | None:
    """Register an external buffer with NVSHMEM and return the mapped address."""
    lib = _get_nvshmem_host_lib()
    res = lib.nvshmemx_buffer_register_symmetric(_ctypes.c_void_p(ptr), size, flags)
    return int(res) if res else None


def buffer_unregister_symmetric(ptr: int, size: int) -> None:
    """Unregister an external buffer previously registered with NVSHMEM."""
    lib = _get_nvshmem_host_lib()
    lib.nvshmemx_buffer_unregister_symmetric(_ctypes.c_void_p(ptr), size)


# Define what gets exposed when users do `import nvshmem.bindings`
__all__ = [name for name in dir() if not name.startswith("_")]
