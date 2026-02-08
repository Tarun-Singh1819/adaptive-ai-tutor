from pathlib import Path
from sentence_transformers import SentenceTransformer
from create_pipeline.normalize_name import normalize_name
from db.sqlite_db import insert_paper
from create_pipeline.cleaning import clean_markdown
from create_pipeline.section_parent_chunk import section_parent_chunk
from create_pipeline.embed_child import embed_children
from create_pipeline.load_and_save import save_docs,save_faiss_index,save_parents
from adaptive_tutor.models import EMBED_MODEL
import numpy as np
import faiss

RAG_ROOT = "rag_store"
def initialize_rag(paper_title: str):
    """
    Fetch paper → clean → chunk → embed → FAISS index → save
    """

    paper_id = normalize_name(paper_title)
    store_path = Path(RAG_ROOT) / paper_id
    store_path.mkdir(parents=True, exist_ok=True)

    # 1. Fetch paper (markdown)
   # md_file = fetch_paper_from_arxiv(paper_title)
    #if md_file is None:
    #    raise ValueError("Paper not found")

    md_path = Path("/Users/tarunsingh/Desktop/rsrch_tutor/Ai_agents.md")
    raw_md = md_path.read_text(encoding="utf-8")

    # 2. Clean
    clean_md = clean_markdown(raw_md)

    # 3. Chunk (parent → children)
    parents = section_parent_chunk(clean_md)
    save_parents(parents, str(store_path))
    # 4. Embeddings
   
    docs = embed_children(parents=parents, model=EMBED_MODEL)

    # 5. FAISS index
    dim = len(docs[0]["embedding"])
    index = faiss.IndexFlatIP(dim)
    index.add(
        np.vstack([d["embedding"] for d in docs]).astype("float32")
    )

    # 6. Save
    save_faiss_index(index, str(store_path))
    save_docs(docs, str(store_path))


    insert_paper(
        paper_id=paper_id,
        title=paper_title,
        rag_path=str(store_path),
        embedding_model="all-MiniLM-L6-v2"
    )

    return str(store_path)
