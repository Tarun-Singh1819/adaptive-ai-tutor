import faiss
import numpy as np

def build_faiss_index(docs):
    dim = len(docs[0]["embedding"])
    index = faiss.IndexFlatIP(dim)  # cosine (normalized embeddings)

    embeddings = np.vstack([d["embedding"] for d in docs]).astype("float32")
    index.add(embeddings)

    return index
