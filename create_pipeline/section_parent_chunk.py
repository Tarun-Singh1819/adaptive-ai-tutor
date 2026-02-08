from dataclasses import dataclass
from typing import List
from core.parent_chunk import ParentChunk

def section_parent_chunk(md_text: str) -> List[ParentChunk]:
    sections = []
    current_title = "Introduction"
    buffer = []

    for line in md_text.splitlines():
        if line.startswith("#"):
            if buffer:
                sections.append((current_title, "\n".join(buffer)))
                buffer = []
            current_title = line.lstrip("#").strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_title, "\n".join(buffer)))

    parents = []
    for idx, (title, content) in enumerate(sections):
        children = [
            p.strip()
            for p in content.split("\n\n")
            if len(p.strip()) > 200
        ]

        parents.append(
            ParentChunk(
                id=f"sec_{idx}",
                title=title,
                content=content,
                children=children
            )
        )

    return parents
