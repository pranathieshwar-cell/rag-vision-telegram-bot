from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Settings
from llm import generate_with_openai
from utils import LRUCache, chunk_text, ensure_dir, iter_text_files


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype("float32")


class RAGStore:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray = np.empty((0, 1), dtype="float32")

    def ensure_index(self) -> None:
        if not self.settings.index_path.exists():
            self.build_index()
        self.load_index()

    def build_index(self) -> None:
        docs_path = self.settings.docs_path
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs path does not exist: {docs_path}")

        docs = list(iter_text_files(docs_path))
        if not docs:
            raise FileNotFoundError(f"No .md or .txt files found in {docs_path}")

        chunks: list[Chunk] = []
        for doc_path in docs:
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
            for idx, chunk in enumerate(
                chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
            ):
                chunks.append(Chunk(doc_id=doc_path.name, chunk_id=idx, text=chunk))

        embeddings = self.embedder.embed([chunk.text for chunk in chunks])

        ensure_dir(self.settings.index_path.parent)
        with sqlite3.connect(self.settings.index_path) as conn:
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute(
                """
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            rows = [
                (chunk.doc_id, chunk.chunk_id, chunk.text, embeddings[i].tobytes())
                for i, chunk in enumerate(chunks)
            ]
            conn.executemany(
                "INSERT INTO chunks (doc_id, chunk_id, text, embedding) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    def load_index(self) -> None:
        if not self.settings.index_path.exists():
            return

        with sqlite3.connect(self.settings.index_path) as conn:
            rows = conn.execute(
                "SELECT doc_id, chunk_id, text, embedding FROM chunks ORDER BY id"
            ).fetchall()

        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        for doc_id, chunk_id, text, embedding in rows:
            chunks.append(Chunk(doc_id=str(doc_id), chunk_id=int(chunk_id), text=str(text)))
            vectors.append(np.frombuffer(embedding, dtype="float32"))

        self._chunks = chunks
        if vectors:
            self._embeddings = np.vstack(vectors)
        else:
            self._embeddings = np.empty((0, 1), dtype="float32")

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._embeddings.size == 0:
            return []

        query_vec = self.embedder.embed([query])[0]
        scores = self._embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = Embedder(settings.embed_model)
        self.store = RAGStore(settings, self.embedder)
        self.store.ensure_index()
        self.cache = LRUCache(settings.cache_size)

    def answer(self, question: str, history: list[tuple[str, str]] | None = None) -> tuple[str, list[str]]:
        cached = self.cache.get(question)
        if cached:
            return cached, []

        results = self.store.search(question, self.settings.top_k)
        prompt = self._build_prompt(question, results, history or [])
        response = generate_with_openai(prompt, self.settings)

        sources = []
        if self.settings.show_sources:
            for result in results:
                snippet = result.chunk.text.strip().replace("\n", " ")
                sources.append(f"{result.chunk.doc_id} :: {snippet[:200]}")

        self.cache.put(question, response)
        return response, sources

    def _build_prompt(
        self, question: str, results: list[SearchResult], history: list[tuple[str, str]]
    ) -> str:
        lines: list[str] = [
            "You are a helpful assistant.",
            "Use the provided context to answer the question.",
            "If the context is insufficient, say you do not know.",
        ]

        if history:
            lines.append("Recent conversation:")
            for q, a in history:
                lines.append(f"Q: {q}")
                lines.append(f"A: {a}")

        if results:
            lines.append("Context:")
            for idx, result in enumerate(results, start=1):
                lines.append(f"[{idx}] {result.chunk.text}")

        lines.append(f"Question: {question}")
        lines.append("Answer:")
        return "\n".join(lines)
