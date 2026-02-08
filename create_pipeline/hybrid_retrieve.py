from create_pipeline.dense_retrieve import  dense_retrieve
from create_pipeline.sparse_retrieve import sparse_retrieve

def hybrid_retrieve(
    query,
    model,
    index,
    docs,
    bm25,
    k_dense=8,
    k_sparse=8,
    k_final=8
):
    dense_hits = dense_retrieve(query, model, index, docs, k_dense)
    sparse_hits = sparse_retrieve(query, bm25, docs, k_sparse)

    seen = set()
    merged = []

    for d in dense_hits + sparse_hits:
        key = d["text"][:80]   # simple dedupe key
        if key not in seen:
            merged.append(d)
            seen.add(key)

        if len(merged) >= k_final:
            break

    return merged
