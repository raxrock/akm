"""
Adaptive Knowledge Mesh (AKM) Framework

An open-source framework combining knowledge graphs, vector databases,
and graph neural networks for adaptive knowledge management.

Example usage:

    from akm import AKM

    # Initialize with default config
    akm = AKM()
    akm.connect()

    # Or with custom config
    akm = AKM.from_config("./akm.yaml")

    # Add entities
    entity = akm.add_entity(
        name="UserService",
        entity_type="Class",
        properties={"language": "python"}
    )

    # Create relationships
    akm.add_relationship(
        source_id=str(entity.id),
        target_id=str(other_entity.id),
        relationship_type="DEPENDS_ON"
    )

    # Create soft links (adaptive)
    link = akm.create_soft_link(
        source_id=str(entity.id),
        target_id=str(other.id),
        pattern_source="co_occurrence"
    )

    # Validate links (user interaction)
    akm.validate_link(str(link.id), is_positive=True)

    # Query (if query engine configured)
    result = akm.query("What are the main architectural patterns?")
"""

__version__ = "0.1.0"

from akm.api.client import AKM
from akm.core.config import AKMConfig
from akm.core.exceptions import (
    AKMError,
    ConfigurationError,
    ConnectionError,
    EntityNotFoundError,
    LinkNotFoundError,
    ValidationError,
)
from akm.core.models import (
    Chunk,
    Document,
    Entity,
    Link,
    LinkStatus,
    LinkWeight,
    QueryResult,
    Relationship,
    SearchResult,
    TraversalResult,
)

# Optional imports for advanced usage
try:
    from akm.gnn import GNNManager
    from akm.ingestion import IngestionPipeline
except ImportError:
    # These may not be available if optional dependencies aren't installed
    GNNManager = None  # type: ignore
    IngestionPipeline = None  # type: ignore

__all__ = [
    # Version
    "__version__",
    # Main client
    "AKM",
    # Models
    "Entity",
    "Relationship",
    "Link",
    "LinkWeight",
    "LinkStatus",
    "Document",
    "Chunk",
    "SearchResult",
    "QueryResult",
    "TraversalResult",
    # Config
    "AKMConfig",
    # Exceptions
    "AKMError",
    "ConfigurationError",
    "ConnectionError",
    "EntityNotFoundError",
    "LinkNotFoundError",
    "ValidationError",
    # Advanced components (optional)
    "GNNManager",
    "IngestionPipeline",
]
