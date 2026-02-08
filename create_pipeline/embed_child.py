def embed_children(parents, model, batch_size=32):
    import numpy as np

    docs = []
    texts = []
    meta = []

    for parent in parents:
        for child in parent.children:
            texts.append(child)
            meta.append({
                "parent_id": parent.id,
                "parent_title": parent.title,
                "parent_content": parent.content,
                "text": child
            })

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for emb, m in zip(embeddings, meta):
        docs.append({
            "text": m["text"],
            "embedding": emb.astype("float32"),
            "parent_id": m["parent_id"],
            "parent_title": m["parent_title"],
            "parent_content": m["parent_content"]
        })

    return docs
