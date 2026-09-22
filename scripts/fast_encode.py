"""Optional compiled preparation of the existing lossless cell record format."""
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


if njit:
    @njit(cache=True)
    def _prepare(indptr, indices, bits, tokens, rows, offsets):
        output = np.empty(offsets[-1], dtype=np.uint8)
        for i in range(len(rows)):
            row = rows[i]
            start, end = indptr[row], indptr[row + 1]
            n = end - start
            base = offsets[i]
            for byte in range(4):
                output[base + byte] = (n >> (8 * byte)) & 255
            previous = 0
            for j in range(n):
                token = int(tokens[indices[start + j]])
                delta = token - previous
                previous = token
                output[base + 4 + j] = delta & 255
                output[base + 4 + n + j] = (delta >> 8) & 255
                value = bits[start + j]
                for byte in range(4):
                    output[base + 4 + 2*n + byte*n + j] = (value >> (8*byte)) & 255
        return output


def prepare_cells(matrix, tokens, rows):
    """Return raw packed bytes plus offsets, or None without Numba.

    Validation and canonical sorting happen once for the whole matrix, rather
    than repeating NumPy allocations and checks for every cell.
    """
    if njit is None:
        return None
    data = matrix.data
    if data.dtype != np.dtype("float32"):
        converted = data.astype(np.float32)
        if not np.array_equal(data, converted):
            raise ValueError("Counts cannot be represented losslessly as float32")
        data = converted
    if not np.all(np.isfinite(data)) or np.any(data <= 0):
        raise ValueError("Counts must be finite and positive")
    if np.any(tokens < 2) or np.any(tokens[1:] <= tokens[:-1]):
        raise ValueError("Compiled encoding requires sorted, unique gene tokens")
    lengths = matrix.indptr[rows + 1] - matrix.indptr[rows]
    if np.any(lengths == 0):
        raise ValueError("Cell has no expressed genes")
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths.astype(np.int64) * 6 + 4, out=offsets[1:])
    raw = _prepare(matrix.indptr, matrix.indices, data.view(np.uint32), tokens, rows, offsets)
    return raw, offsets
