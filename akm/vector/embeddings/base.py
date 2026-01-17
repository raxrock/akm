"""Base embedding model and factory."""

from __future__ import annotations

from typing import Union

from akm.core.config import EmbeddingConfig
from akm.core.exceptions import ConfigurationError
from akm.core.interfaces import EmbeddingModel


def create_embedding_model(config: Union[EmbeddingConfig, dict]) -> EmbeddingModel:
    """
    Create an embedding model based on configuration.

    Args:
        config: Embedding configuration object or dict

    Returns:
        Configured EmbeddingModel instance

    Raises:
        ConfigurationError: If provider is not supported
    """
    if isinstance(config, dict):
        config = EmbeddingConfig(**config)

    provider = config.provider.lower()

    if provider == "sentence_transformers":
        from akm.vector.embeddings.sentence_transformers import (
            SentenceTransformerEmbedding,
        )

        return SentenceTransformerEmbedding(config.sentence_transformers)
    elif provider == "openai":
        from akm.vector.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(config.openai)
    else:
        raise ConfigurationError(
            f"Unsupported embedding provider: {provider}",
            details={"supported_providers": ["sentence_transformers", "openai"]},
        )


__all__ = ["create_embedding_model"]
