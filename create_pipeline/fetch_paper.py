import os
import arxiv
import requests
import time
from pathlib import Path


def fetch_paper_from_arxiv(paper_title: str, download_dir: str = "./data/papers"):
    """
    ArXiv API ka use karke paper search karega aur PDF download karega.
    """
    # Ensure directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Add delay to respect rate limits
    time.sleep(3)

    # 1. Search logic using Client API (not deprecated Search.results())
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"ti:{paper_title}", # 'ti' search specifically in the title
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = list(client.results(search))
    
    if not results:
        print(f"Paper '{paper_title}' not found on ArXiv.")
        return None

    paper = results[0]
    # File name ko clean karna zaroori hai (removing spaces/special chars)
    file_name = f"{paper_title.replace(' ', '_').lower()}.pdf"
    file_path = os.path.join(download_dir, file_name)

    # 2. Download logic
    print(f"Downloading: {paper.title}...")
    try:
        paper.download_pdf(dirpath=download_dir, filename=file_name)
        print(f"Saved to: {file_path}")
        
        # Return metadata for the Vector DB
        return {
            "status": "success",
            "file_path": file_path,
            "title": paper.title,
            "entry_id": paper.entry_id,
            "summary": paper.summary
        }
    except Exception as e:
        print(f"Download failed: {e}")
        return None