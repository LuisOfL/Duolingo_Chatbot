import numpy as np

class IVFIndexFromScratch:
    def __init__(self, nlist: int = 10, nprobe: int = 2):
        """
        :param nlist: Número de clusters (listas invertidas/centroides).
        :param nprobe: Número de clusters a inspeccionar durante la búsqueda.
        """
        self.nlist = nlist
        self.nprobe = nprobe
        self.centroids = None
        # Diccionario para almacenar las listas invertidas: { centroid_id: [(chunk_id, vector), ...] }
        self.inverted_lists = {i: [] for i in range(nlist)}

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calcula la distancia de coseno entre un conjunto de vectores y otro."""
        norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
        # Evitar división por cero
        norm_a[norm_a == 0] = 1e-10
        norm_b[norm_b == 0] = 1e-10
        
        similitud = np.dot(a, b.T) / (norm_a * norm_b.T)
        return 1.0 - similitud

    def fit(self, vectors: np.ndarray, max_iters: int = 20):
        """Entrena los centroides mediante K-Means usando distancia coseno."""
        num_samples, dim = vectors.shape
        if num_samples < self.nlist:
            raise ValueError("El número de muestras debe ser mayor o igual a nlist.")

        # Inicialización aleatoria de centroides
        indices = np.random.choice(num_samples, self.nlist, replace=False)
        self.centroids = vectors[indices].copy()

        for _ in range(max_iters):
            # Asignar cada vector al centroide más cercano
            distances = np.array([self._cosine_distance(vectors, np.array([c])).flatten() for c in self.centroids]).T
            cluster_assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.nlist):
                members = vectors[cluster_assignments == i]
                if len(members) > 0:
                    new_centroids[i] = np.mean(members, axis=0)
                else:
                    new_centroids[i] = vectors[np.random.choice(num_samples)]

            if np.allclose(self.centroids, new_centroids, atol=1e-5):
                break
            self.centroids = new_centroids

    def add(self, chunk_ids: list, vectors: np.ndarray):
        """Asigna los vectores a las listas invertidas según el centroide más cercano."""
        if self.centroids is None:
            raise RuntimeError("El índice debe ser entrenado con .fit() antes de agregar vectores.")

        # Buscar el centroide más cercano para cada vector
        for chunk_id, vec in zip(chunk_ids, vectors):
            distances = self._cosine_distance(np.array([vec]), self.centroids).flatten()
            closest_centroid = int(np.argmin(distances))
            self.inverted_lists[closest_centroid].append((chunk_id, vec))

    def search(self, query_vector: np.ndarray, k: int = 3) -> list:
        """
        Busca los k vectores más cercanos filtrando solo en las nprobe listas invertidas más cercanas.
        """
        query_vec = np.array([query_vector])
        
        # 1. Encontrar los nprobe centroides más cercanos a la consulta
        centroid_distances = self._cosine_distance(query_vec, self.centroids).flatten()
        closest_centroids = np.argsort(centroid_distances)[:self.nprobe]

        candidatos = []
        # 2. Escanear únicamente los elementos en esas listas invertidas
        for centroid_idx in closest_centroids:
            for chunk_id, vec in self.inverted_lists[centroid_idx]:
                dist = float(self._cosine_distance(query_vec, np.array([vec]))[0, 0])
                candidatos.append((chunk_id, dist))

        # 3. Ordenar los candidatos recuperados por distancia y devolver los k mejores
        candidatos.sort(key=lambda x: x[1])
        return candidatos[:k]