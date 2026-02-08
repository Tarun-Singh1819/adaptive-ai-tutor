# core/parent_chunk.py
from dataclasses import dataclass
from typing import List

@dataclass
class ParentChunk:
    id: str
    title: str
    content: str
    children: List[str]
