from collections import OrderedDict

MAX_PAPERS_IN_CACHE = 2

# paper_id -> {index, docs, parents, bm25}
RAG_CACHE = OrderedDict()
