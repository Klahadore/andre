"""Optional libdeflate acceleration for gzip-only, one-dimensional HDF5 arrays.

All unsupported layouts/filters use h5py's normal decoder. The file format is
unchanged. Each build worker owns its decoder; no threads share its state.
"""
import ctypes as C
import ctypes.util
import numpy as np

_library = ctypes.util.find_library("deflate")
_lib = C.CDLL(_library) if _library else None
_decoder = None
if _lib:
    _lib.libdeflate_alloc_decompressor.restype = C.c_void_p
    _lib.libdeflate_zlib_decompress.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t,
                                              C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    _lib.libdeflate_zlib_decompress.restype = C.c_int
    _decoder = _lib.libdeflate_alloc_decompressor()


def read_vector(ds):
    plist = ds.id.get_create_plist()
    if (not _decoder or not ds.chunks or len(ds.shape) != 1
            or plist.get_nfilters() != 1 or plist.get_filter(0)[0] != 1):
        return ds[:]
    chunk = ds.chunks[0]
    chunks = (ds.size + chunk - 1) // chunk
    output = np.empty(chunks * chunk, dtype=ds.dtype)
    chunk_bytes = chunk * ds.dtype.itemsize
    actual = C.c_size_t()
    for j in range(chunks):
        mask, compressed = ds.id.read_direct_chunk((j * chunk,))
        if mask:
            return ds[:]
        status = _lib.libdeflate_zlib_decompress(
            _decoder, compressed, len(compressed), output.ctypes.data + j * chunk_bytes,
            chunk_bytes, C.byref(actual),
        )
        if status != 0 or actual.value != chunk_bytes:
            raise ValueError(f"Invalid gzip chunk in {ds.name}: status={status}, bytes={actual.value}")
    return output[:ds.size]
