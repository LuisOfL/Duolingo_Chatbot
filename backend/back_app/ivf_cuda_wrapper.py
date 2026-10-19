import ctypes
import os
import platform
import numpy as np

# Cargar la librería dinámica según el sistema operativo
lib_name = "ivf_cuda.dll" if platform.system() == "Windows" else "./libivf_cuda.so"
lib_path = os.path.abspath(lib_name)

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"No se encontró la librería compilada de CUDA en {lib_path}")

cuda_lib = ctypes.CDLL(lib_path)

# Definir la estructura opaca del índice
class IVFIndexCUDA(ctypes.Structure):
    pass

# Tipado de funciones C
cuda_lib.build_ivf_cuda.argtypes = [
    ctypes.POINTER(ctypes.c_float), # h_vectors
    ctypes.POINTER(ctypes.c_int),   # h_ids
    ctypes.c_int,                   # N
    ctypes.c_int,                   # dim
    ctypes.c_int,                   # nlist
    ctypes.c_int,                   # nprobe
    ctypes.c_int                    # max_iters
]
cuda_lib.build_ivf_cuda.restype = ctypes.POINTER(IVFIndexCUDA)

cuda_lib.search_ivf_cuda.argtypes = [
    ctypes.POINTER(IVFIndexCUDA),    # index
    ctypes.POINTER(ctypes.c_float), # h_query
    ctypes.c_int,                   # k
    ctypes.POINTER(ctypes.c_int),   # out_ids
    ctypes.POINTER(ctypes.c_float)  # out_distances
]
cuda_lib.search_ivf_cuda.restype = ctypes.c_int

cuda_lib.free_ivf_cuda.argtypes = [ctypes.POINTER(IVFIndexCUDA)]
cuda_lib.free_ivf_cuda.restype = None


class IVFIndexCUDAPython:
    def __init__(self, nlist: int = 10, nprobe: int = 2, max_iters: int = 20):
        self.nlist = nlist
        self.nprobe = nprobe
        self.max_iters = max_iters
        self.index_ptr = None

    def build(self, chunk_ids: list, vectors: np.ndarray):
        N, dim = vectors.shape
        vectors_flat = vectors.astype(np.float32).flatten()
        ids_arr = np.array(chunk_ids, dtype=np.int32)

        c_vectors = vectors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ids = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.index_ptr = cuda_lib.build_ivf_cuda(
            c_vectors, c_ids, N, dim, self.nlist, self.nprobe, self.max_iters
        )

    def search(self, query_vec: np.ndarray, k: int = 3):
        if not self.index_ptr:
            raise RuntimeError("El índice IVF en CUDA no se ha inicializado.")

        query_flat = query_vec.astype(np.float32).flatten()
        c_query = query_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        out_ids = (ctypes.c_int * k)()
        out_dists = (ctypes.c_float * k)()

        num_found = cuda_lib.search_ivf_cuda(self.index_ptr, c_query, k, out_ids, out_dists)

        resultados = [(out_ids[i], float(out_dists[i])) for i in range(num_found)]
        return resultados

    def __del__(self):
        if self.index_ptr:
            cuda_lib.free_ivf_cuda(self.index_ptr)