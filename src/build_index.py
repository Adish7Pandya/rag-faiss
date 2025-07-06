import faiss
from embedder import Embedder

def build_faiss_index(docs, index_path="faiss.index"):
    embedder = Embedder()
    vectors = embedder.encode(docs)
    
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    faiss.write_index(index, index_path)
    return index, vectors
