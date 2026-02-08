from rank_bm25 import BM25Okapi

def build_bm25(docs):
    corpus = [d["text"] for d in docs]
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus
