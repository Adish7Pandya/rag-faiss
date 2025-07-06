import os

with open("data/sample_docs.txt") as f:
    docs = [line.strip() for line in f if line.strip()]

if not os.path.exists("faiss.index"):
    from src.build_index import build_faiss_index
    build_faiss_index(docs)

from src.retriever import Retriever
from src.rag import RAG

retriever = Retriever(docs)
rag = RAG()

question = "Where is the Eiffel Tower?"
contexts = retriever.query(question, top_k=2)

print("Retrieved Contexts:", contexts)

answer = rag.generate(question, contexts)
print("Generated Answer:", answer)
