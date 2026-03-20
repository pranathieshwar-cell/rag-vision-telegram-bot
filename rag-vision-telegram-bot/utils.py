from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
from typing import Iterable


class LRUCache:
    def __init__(self, capacity: int = 128) -> None:
        self.capacity = max(1, capacity)
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            yield path


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunk_size = max(20, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def extract_tags(text: str, k: int = 3) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
    tags: list[str] = []
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if token not in tags:
            tags.append(token)
        if len(tags) >= k:
            break
    return tags
