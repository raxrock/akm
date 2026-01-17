"""Graph backend base and factory functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from akm.core.config import GraphConfig
from akm.core.exceptions import ConfigurationError
from akm.core.interfaces import GraphBackend

if TYPE_CHECKING:
    from akm.core.models import Entity, Relationship


def create_graph_backend(
    config: Union[GraphConfig, dict]
) -> GraphBackend["Entity", "Relationship"]:
    """
    Create a graph backend based on configuration.

    Args:
        config: Graph configuration object or dict

    Returns:
        Configured GraphBackend instance

    Raises:
        ConfigurationError: If backend type is not supported
    """
    if isinstance(config, dict):
        config = GraphConfig(**config)

    backend_type = config.backend.lower()

    if backend_type == "neo4j":
        from akm.graph.neo4j.client import Neo4jBackend

        return Neo4jBackend(config.neo4j)
    elif backend_type == "memory":
        from akm.graph.memory.client import MemoryGraphBackend

        return MemoryGraphBackend(config.memory)
    else:
        raise ConfigurationError(
            f"Unsupported graph backend: {backend_type}",
            details={"supported_backends": ["neo4j", "memory"]},
        )


__all__ = ["create_graph_backend"]
