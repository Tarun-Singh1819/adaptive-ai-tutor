def sparse_retrieve(query, bm25, docs, k=8):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [docs[i] for i in top_idx]
