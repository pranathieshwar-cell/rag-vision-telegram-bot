from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency
    load_dotenv = None

if load_dotenv is not None:
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path)


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    mode: str

    docs_path: Path
    index_path: Path
    image_dir: Path

    chunk_size: int
    chunk_overlap: int
    top_k: int
    embed_model: str

    openai_api_key: str
    openai_model: str

    show_sources: bool
    enable_history: bool
    history_len: int
    cache_size: int

    vision_model: str


def load_settings() -> Settings:
    root = Path(__file__).resolve().parent
    return Settings(
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        mode=os.getenv("MODE", "hybrid").lower(),
        docs_path=Path(os.getenv("DOCS_PATH", str(root / "data" / "docs"))),
        index_path=Path(os.getenv("INDEX_PATH", str(root / "data" / "index.sqlite"))),
        image_dir=Path(os.getenv("IMAGE_DIR", str(root / "data" / "images"))),
        chunk_size=_get_int("CHUNK_SIZE", 400),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 80),
        top_k=_get_int("TOP_K", 3),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5"),
        show_sources=_get_bool("SHOW_SOURCES", False),
        enable_history=_get_bool("ENABLE_HISTORY", True),
        history_len=_get_int("HISTORY_LEN", 3),
        cache_size=_get_int("CACHE_SIZE", 128),
        vision_model=os.getenv("VISION_MODEL", "Salesforce/blip-image-captioning-base"),
    )
