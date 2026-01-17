"""Sentence Transformers embedding model implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import List

from akm.core.config import SentenceTransformersConfig
from akm.core.exceptions import EmbeddingError
from akm.core.interfaces import EmbeddingModel

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using Sentence Transformers."""

    def __init__(self, config: SentenceTransformersConfig) -> None:
        """
        Initialize Sentence Transformer embedding model.

        Args:
            config: Sentence Transformers configuration
        """
        self._config = config
        self._model = None
        self._dimension: int = 0

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._config.model_name,
                device=self._config.device,
            )
            # Get dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded SentenceTransformer model: {self._config.model_name} "
                f"(dim={self._dimension})"
            )
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension == 0:
            self._load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._config.model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self._load_model()

        if not texts:
            return []

        try:
            # Process in batches
            all_embeddings = []
            batch_size = self._config.batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                all_embeddings.extend(embeddings.tolist())

            return all_embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of embed.

        Note: Sentence Transformers doesn't have native async support,
        so this runs in a thread pool.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, texts)
