import os
import json
import boto3
import psycopg2
import numpy as np
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from ivf_cuda_wrapper import IVFIndexCUDAPython  # Importación del módulo CUDA

load_dotenv()

# Instancia global del índice IVF en CUDA
ivf_cuda_index = None

def inicializar_o_obtener_indice_cuda(cursor) -> IVFIndexCUDAPython:
    """
    Extrae los vectores de RDS PostgreSQL y construye/entrena el índice IVF en la GPU usando CUDA.
    """
    global ivf_cuda_index
    if ivf_cuda_index is not None:
        return ivf_cuda_index

    cursor.execute("SELECT id, embedding FROM public.duolingo_chunks;")
    filas = cursor.fetchall()

    if not filas:
        return None

    chunk_ids = [fila[0] for fila in filas]
    vectores = np.array([json.loads(fila[1]) if isinstance(fila[1], str) else fila[1] for fila in filas], dtype=np.float32)

    nlist = min(10, len(chunk_ids))
    
    # Inicializar y entrenar el índice en GPU mediante la DLL/SO
    ivf_cuda_index = IVFIndexCUDAPython(nlist=nlist, nprobe=2, max_iters=15)
    ivf_cuda_index.build(chunk_ids, vectores)

    print("🚀 ¡Índice IVF acelerado con CUDA construido exitosamente!")
    return ivf_cuda_index


def buscar_chunks_cercanos(vector_consulta: list, k: int = 3) -> list:
    """
    Utiliza el índice IVF en CUDA para obtener los IDs más cercanos
    y recupera el contenido textual de PostgreSQL.
    """
    if not vector_consulta:
        print("El vector de consulta está vacío.")
        return []

    connection = None
    cursor = None

    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database="vectorial-rag",
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
        cursor = connection.cursor()

        # 1. Obtener o construir el índice IVF acelerado por CUDA
        index_cuda = inicializar_o_obtener_indice_cuda(cursor)
        if index_cuda is None:
            return []

        # 2. Ejecutar búsqueda ANN en GPU
        query_vec = np.array(vector_consulta, dtype=np.float32)
        resultados_cuda = index_cuda.search(query_vec, k=k)

        if not resultados_cuda:
            return []

        ids_candidatos = [item[0] for item in resultados_cuda]
        distancias_map = {item[0]: item[1] for item in resultados_cuda}

        # 3. Traer los chunks de la BD para las IDs filtradas por la GPU
        sql_query = """
            SELECT id, section_title, source, chunk_text
            FROM public.duolingo_chunks
            WHERE id = ANY(%s);
        """
        cursor.execute(sql_query, (ids_candidatos,))
        filas = cursor.fetchall()

        resultados_cercanos = []
        for fila in filas:
            chunk_id, section_title, source, chunk_text = fila
            resultados_cercanos.append({
                "section_title": section_title,
                "source": source,
                "chunk_text": chunk_text,
                "distancia": distancias_map.get(chunk_id, 0.0)
            })

        # Preservar el ordenamiento determinado por CUDA
        resultados_cercanos.sort(key=lambda x: x["distancia"])

        print(f"¡Búsqueda con CUDA IVF completada! Recuperados {len(resultados_cercanos)} chunks.")
        return resultados_cercanos

    except Exception as e:
        print(f"❌ Error al realizar la búsqueda vectorial con CUDA: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()