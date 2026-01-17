"""OpenAI embedding model implementation."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from akm.core.config import OpenAIEmbeddingConfig
from akm.core.exceptions import EmbeddingError
from akm.core.interfaces import EmbeddingModel

logger = logging.getLogger(__name__)

# Model dimensions for OpenAI embedding models
OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingModel):
    """Embedding model using OpenAI API."""

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        """
        Initialize OpenAI embedding model.

        Args:
            config: OpenAI embedding configuration
        """
        self._config = config
        self._client = None
        self._async_client = None

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide api_key in configuration."
            )
        return api_key

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._get_api_key())
            except ImportError:
                raise EmbeddingError(
                    "openai not installed. Install with: pip install openai"
                )
        return self._client

    async def _get_async_client(self):
        """Get or create async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                self._async_client = AsyncOpenAI(api_key=self._get_api_key())
            except ImportError:
                raise EmbeddingError(
                    "openai not installed. Install with: pip install openai"
                )
        return self._async_client

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return OPENAI_EMBEDDING_DIMENSIONS.get(self._config.model, 1536)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._config.model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = self._get_client()
        all_embeddings = []
        batch_size = self._config.batch_size

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = client.embeddings.create(
                    model=self._config.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate OpenAI embeddings: {e}")

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of embed.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = await self._get_async_client()
        all_embeddings = []
        batch_size = self._config.batch_size

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await client.embeddings.create(
                    model=self._config.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate OpenAI embeddings: {e}")
