"""Vector database module for the AKM framework."""

from akm.vector.base import create_vector_backend
from akm.vector.chromadb.chunking import (
    FixedSizeChunking,
    ParagraphChunking,
    SentenceChunking,
    get_chunking_strategy,
)
from akm.vector.embeddings.base import create_embedding_model

__all__ = [
    "create_vector_backend",
    "create_embedding_model",
    "FixedSizeChunking",
    "SentenceChunking",
    "ParagraphChunking",
    "get_chunking_strategy",
]
