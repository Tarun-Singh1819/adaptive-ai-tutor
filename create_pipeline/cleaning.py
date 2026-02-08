import re

def clean_markdown(md_text: str) -> str:
    # remove references section
    md_text = re.split(r"\n#+\s*references", md_text, flags=re.I)[0]

    # remove figure/table captions
    md_text = re.sub(r"\n\s*(figure|table)\s*\d+.*", "", md_text, flags=re.I)

    # remove inline citations like [12], (Smith et al., 2020)
    md_text = re.sub(r"\[[0-9,\s]+\]", "", md_text)
    md_text = re.sub(r"\([A-Za-z]+ et al\.,?\s*\d{4}\)", "", md_text)

    # collapse excessive newlines
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)

    return md_text.strip()
