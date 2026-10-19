#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Error CUDA en %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel para calcular la distancia de coseno entre N vectores y M centroides en GPU
__global__ void cosine_distance_kernel(const float* vectors, const float* centroids, float* distances, int N, int M, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Índice de vector N
    int col = blockIdx.y * blockDim.y + threadIdx.y; // Índice de centroide M

    if (row < N && col < M) {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (int d = 0; d < dim; ++d) {
            float val_a = vectors[row * dim + d];
            float val_b = centroids[col * dim + d];
            dot += val_a * val_b;
            norm_a += val_a * val_a;
            norm_b += val_b * val_b;
        }

        norm_a = sqrtf(norm_a);
        norm_b = sqrtf(norm_b);

        if (norm_a < 1e-10f) norm_a = 1e-10f;
        if (norm_b < 1e-10f) norm_b = 1e-10f;

        float similarity = dot / (norm_a * norm_b);
        distances[row * M + col] = 1.0f - similarity; // Distancia de Coseno
    }
}

// Estructura C para representar las listas invertidas
typedef struct {
    int nlist;
    int nprobe;
    int dim;
    int total_vectors;
    float* d_centroids; // Centroides guardados en GPU
    int* h_chunk_ids;   // Mapeo de IDs
    float* h_vectors;   // Copia local de los vectores
    int* inverted_list_sizes;
    int** inverted_lists_ids;
    float** inverted_lists_vecs;
} IVFIndexCUDA;

extern "C" {

// 1. Crear y entrenar el índice IVF en CUDA
IVFIndexCUDA* build_ivf_cuda(const float* h_vectors, const int* h_ids, int N, int dim, int nlist, int nprobe, int max_iters) {
    IVFIndexCUDA* index = (IVFIndexCUDA*)malloc(sizeof(IVFIndexCUDA));
    index->nlist = nlist;
    index->nprobe = nprobe;
    index->dim = dim;
    index->total_vectors = N;

    // Memoria en Host
    index->h_chunk_ids = (int*)malloc(N * sizeof(int));
    index->h_vectors = (float*)malloc(N * dim * sizeof(float));
    memcpy(index->h_chunk_ids, h_ids, N * sizeof(int));
    memcpy(index->h_vectors, h_vectors, N * dim * sizeof(float));

    // Memoria en Device (GPU)
    float *d_vectors, *d_centroids, *d_distances;
    CUDA_CHECK(cudaMalloc((void**)&d_vectors, N * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, nlist * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_distances, N * nlist * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, N * dim * sizeof(float), cudaMemcpyHostToDevice));

    // Inicializar centroides seleccionando NList puntos aleatorios
    float* h_centroids = (float*)malloc(nlist * dim * sizeof(float));
    for (int i = 0; i < nlist; ++i) {
        int rand_idx = rand() % N;
        memcpy(&h_centroids[i * dim], &h_vectors[rand_idx * dim], dim * sizeof(float));
    }
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, nlist * dim * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (nlist + blockSize.y - 1) / blockSize.y);

    // K-Means Training Loop en GPU
    float* h_distances = (float*)malloc(N * nlist * sizeof(float));
    int* assignments = (int*)malloc(N * sizeof(int));
    int* counts = (int*)malloc(nlist * sizeof(int));

    for (int iter = 0; iter < max_iters; ++iter) {
        cosine_distance_kernel<<<gridSize, blockSize>>>(d_vectors, d_centroids, d_distances, N, nlist, dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_distances, d_distances, N * nlist * sizeof(float), cudaMemcpyDeviceToHost));

        // Reasignar clusters en CPU
        for (int i = 0; i < N; ++i) {
            int min_idx = 0;
            float min_dist = h_distances[i * nlist];
            for (int j = 1; j < nlist; ++j) {
                if (h_distances[i * nlist + j] < min_dist) {
                    min_dist = h_distances[i * nlist + j];
                    min_idx = j;
                }
            }
            assignments[i] = min_idx;
        }

        // Recalcular centroides
        memset(h_centroids, 0, nlist * dim * sizeof(float));
        memset(counts, 0, nlist * sizeof(int));

        for (int i = 0; i < N; ++i) {
            int c = assignments[i];
            counts[c]++;
            for (int d = 0; d < dim; ++d) {
                h_centroids[c * dim + d] += h_vectors[i * dim + d];
            }
        }

        for (int c = 0; c < nlist; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < dim; ++d) {
                    h_centroids[c * dim + d] /= counts[c];
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, nlist * dim * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Construir Listas Invertidas
    index->inverted_list_sizes = (int*)calloc(nlist, sizeof(int));
    for (int i = 0; i < N; ++i) {
        index->inverted_list_sizes[assignments[i]]++;
    }

    index->inverted_lists_ids = (int**)malloc(nlist * sizeof(int*));
    index->inverted_lists_vecs = (float**)malloc(nlist * sizeof(float*));

    for (int c = 0; c < nlist; ++c) {
        int sz = index->inverted_list_sizes[c];
        index->inverted_lists_ids[c] = (int*)malloc(sz * sizeof(int));
        index->inverted_lists_vecs[c] = (float*)malloc(sz * dim * sizeof(float));
    }

    int* current_pos = (int*)calloc(nlist, sizeof(int));
    for (int i = 0; i < N; ++i) {
        int c = assignments[i];
        int pos = current_pos[c];
        index->inverted_lists_ids[c][pos] = h_ids[i];
        memcpy(&index->inverted_lists_vecs[c][pos * dim], &h_vectors[i * dim], dim * sizeof(float));
        current_pos[c]++;
    }

    index->d_centroids = d_centroids;

    // Limpieza auxiliar
    free(h_centroids);
    free(h_distances);
    free(assignments);
    free(counts);
    free(current_pos);
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_distances));

    return index;
}

// 2. Búsqueda ANN en CUDA inspeccionando los nprobe centroides más cercanos
int search_ivf_cuda(IVFIndexCUDA* index, const float* h_query, int k, int* out_ids, float* out_distances) {
    int dim = index->dim;
    int nlist = index->nlist;

    float *d_query, *d_distances;
    CUDA_CHECK(cudaMalloc((void**)&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_distances, nlist * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_query, h_query, dim * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(1, 16);
    dim3 gridSize(1, (nlist + blockSize.y - 1) / blockSize.y);

    // Calcular distancia del vector de consulta a los centroides usando CUDA
    cosine_distance_kernel<<<gridSize, blockSize>>>(d_query, index->d_centroids, d_distances, 1, nlist, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_centroid_dists = (float*)malloc(nlist * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_centroid_dists, d_distances, nlist * sizeof(float), cudaMemcpyDeviceToHost));

    // Seleccionar los nprobe centroides más cercanos
    int* nearest_centroids = (int*)malloc(nlist * sizeof(int));
    for (int i = 0; i < nlist; ++i) nearest_centroids[i] = i;

    for (int i = 0; i < nlist - 1; ++i) {
        for (int j = i + 1; j < nlist; ++j) {
            if (h_centroid_dists[nearest_centroids[i]] > h_centroid_dists[nearest_centroids[j]]) {
                int temp = nearest_centroids[i];
                nearest_centroids[i] = nearest_centroids[j];
                nearest_centroids[j] = temp;
            }
        }
    }

    // Recolectar candidatos en las listas invertidas de los nprobe centroides
    int max_candidatos = 0;
    for (int p = 0; p < index->nprobe; ++p) {
        int c = nearest_centroids[p];
        max_candidatos += index->inverted_list_sizes[c];
    }

    int* candidate_ids = (int*)malloc(max_candidatos * sizeof(int));
    float* candidate_dists = (float*)malloc(max_candidatos * sizeof(float));
    int cand_count = 0;

    for (int p = 0; p < index->nprobe; ++p) {
        int c = nearest_centroids[p];
        int sz = index->inverted_list_sizes[c];
        for (int i = 0; i < sz; ++i) {
            int cid = index->inverted_lists_ids[c][i];
            float* vec = &index->inverted_lists_vecs[c][i * dim];

            // Calcular distancia de coseno puntual en CPU para los candidatos
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += h_query[d] * vec[d];
                norm_a += h_query[d] * h_query[d];
                norm_b += vec[d] * vec[d];
            }
            float dist = 1.0f - (dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-10f));

            candidate_ids[cand_count] = cid;
            candidate_dists[cand_count] = dist;
            cand_count++;
        }
    }

    // Ordenar candidatos y extraer Top-K
    for (int i = 0; i < cand_count - 1; ++i) {
        for (int j = i + 1; j < cand_count; ++j) {
            if (candidate_dists[i] > candidate_dists[j]) {
                float td = candidate_dists[i];
                candidate_dists[i] = candidate_dists[j];
                candidate_dists[j] = td;

                int ti = candidate_ids[i];
                candidate_ids[i] = candidate_ids[j];
                candidate_ids[j] = ti;
            }
        }
    }

    int returned_k = (cand_count < k) ? cand_count : k;
    for (int i = 0; i < returned_k; ++i) {
        out_ids[i] = candidate_ids[i];
        out_distances[i] = candidate_dists[i];
    }

    // Liberar memoria temporal
    free(h_centroid_dists);
    free(nearest_centroids);
    free(candidate_ids);
    free(candidate_dists);
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_distances));

    return returned_k;
}

// 3. Liberar memoria reservada en el índice
void free_ivf_cuda(IVFIndexCUDA* index) {
    if (!index) return;
    CUDA_CHECK(cudaFree(index->d_centroids));
    for (int c = 0; c < index->nlist; ++c) {
        free(index->inverted_lists_ids[c]);
        free(index->inverted_lists_vecs[c]);
    }
    free(index->inverted_lists_ids);
    free(index->inverted_lists_vecs);
    free(index->inverted_list_sizes);
    free(index->h_chunk_ids);
    free(index->h_vectors);
    free(index);
}

} // extern "C"