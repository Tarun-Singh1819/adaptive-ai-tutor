def dense_retrieve(query, model, index, docs, k=8):
    import numpy as np

    q_emb = model.encode(
        query,
        normalize_embeddings=True
    ).astype("float32")

    _, indices = index.search(q_emb.reshape(1, -1), k)
    return [docs[i] for i in indices[0]]
