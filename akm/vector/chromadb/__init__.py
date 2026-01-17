"""ChromaDB vector backend module."""

from akm.vector.chromadb.chunking import (
    FixedSizeChunking,
    ParagraphChunking,
    SentenceChunking,
    get_chunking_strategy,
)
from akm.vector.chromadb.client import ChromaDBBackend

__all__ = [
    "ChromaDBBackend",
    "FixedSizeChunking",
    "SentenceChunking",
    "ParagraphChunking",
    "get_chunking_strategy",
]
