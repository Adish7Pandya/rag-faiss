import faiss
from embedder import Embedder

class Retriever:
    def __init__(self, docs, index_path="faiss.index"):
        self.docs = docs
        self.embedder = Embedder()
        self.index = faiss.read_index(index_path)
    
    def query(self, text, top_k=3):
        query_vector = self.embedder.encode([text])
        D, I = self.index.search(query_vector, top_k)
        retrieved = [self.docs[idx] for idx in I[0]]
        return retrieved
