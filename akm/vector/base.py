"""Vector backend base and factory functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from akm.core.config import VectorConfig
from akm.core.exceptions import ConfigurationError
from akm.core.interfaces import VectorBackend

if TYPE_CHECKING:
    pass


def create_vector_backend(config: Union[VectorConfig, dict]) -> VectorBackend:
    """
    Create a vector backend based on configuration.

    Args:
        config: Vector configuration object or dict

    Returns:
        Configured VectorBackend instance

    Raises:
        ConfigurationError: If backend type is not supported
    """
    if isinstance(config, dict):
        config = VectorConfig(**config)

    backend_type = config.backend.lower()

    if backend_type == "chromadb":
        from akm.vector.chromadb.client import ChromaDBBackend

        return ChromaDBBackend(config.chromadb)
    else:
        raise ConfigurationError(
            f"Unsupported vector backend: {backend_type}",
            details={"supported_backends": ["chromadb"]},
        )


__all__ = ["create_vector_backend"]
