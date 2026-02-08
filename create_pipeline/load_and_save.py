import faiss
import pickle
import os
import numpy as np
from core.parent_chunk import ParentChunk

def save_faiss_index(index, path: str):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "faiss.index"))


def load_faiss_index(path: str):
    index_path = os.path.join(path, "faiss.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found")

    return faiss.read_index(index_path)


def save_docs(docs, path: str):
    with open(os.path.join(path, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)


def load_docs(path: str):
    with open(os.path.join(path, "docs.pkl"), "rb") as f:
        return pickle.load(f)
    
def save_parents(parents, path: str):
    with open(os.path.join(path, "parents.pkl"), "wb") as f:
        pickle.dump(parents, f)

def load_parents(path: str):
    with open(os.path.join(path, "parents.pkl"), "rb") as f:
        return pickle.load(f)
