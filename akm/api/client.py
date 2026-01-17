"""Main AKM client for programmatic usage."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from akm.adapters import get_domain_transformer
from akm.core.config import AKMConfig, load_config
from akm.core.exceptions import ConfigurationError
from akm.core.interfaces import (
    DomainTransformer,
    EmbeddingModel,
    GraphBackend,
    LLMProvider,
    VectorBackend,
)
from akm.core.models import (
    Document,
    Entity,
    Link,
    LinkStatus,
    QueryResult,
    Relationship,
    SearchResult,
    TraversalResult,
)
from akm.graph.base import create_graph_backend
from akm.links.manager import LinkManager
from akm.query.engine import QueryEngine
from akm.vector.base import create_vector_backend
from akm.vector.embeddings.base import create_embedding_model

logger = logging.getLogger(__name__)


class AKM:
    """
    Main client for the Adaptive Knowledge Mesh framework.

    Example usage:

    ```python
    from akm import AKM

    # Initialize with default config
    akm = AKM()

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

    # Run decay on links
    akm.run_link_decay()

    # Query (if query engine configured)
    result = akm.query("What are the main architectural patterns?")
    ```
    """

    def __init__(
        self,
        config: Optional[AKMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the AKM client.

        Args:
            config: Configuration object (optional)
            config_path: Path to YAML config file (optional)
        """
        if config:
            self._config = config
        elif config_path:
            self._config = load_config(config_path)
        else:
            self._config = AKMConfig()

        self._graph: Optional[GraphBackend] = None
        self._vector: Optional[VectorBackend] = None
        self._embedding: Optional[EmbeddingModel] = None
        self._llm: Optional[LLMProvider] = None
        self._link_manager: Optional[LinkManager] = None
        self._query_engine: Optional[QueryEngine] = None
        self._domain_transformer: Optional[DomainTransformer] = None
        self._initialized = False

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "AKM":
        """Create an AKM instance from a YAML config file."""
        return cls(config_path=path)

    @property
    def config(self) -> AKMConfig:
        """Get the current configuration."""
        return self._config

    @property
    def graph(self) -> GraphBackend:
        """Get the graph backend."""
        if not self._graph:
            raise ConfigurationError("AKM not initialized. Call connect() first.")
        return self._graph

    @property
    def link_manager(self) -> LinkManager:
        """Get the link manager."""
        if not self._link_manager:
            raise ConfigurationError("AKM not initialized. Call connect() first.")
        return self._link_manager

    def connect(self) -> None:
        """
        Connect to all backends and initialize components.

        This must be called before using the AKM client.
        """
        if self._initialized:
            return

        # Initialize graph backend
        self._graph = create_graph_backend(self._config.graph)
        self._graph.connect()

        # Initialize link manager
        self._link_manager = LinkManager(
            graph=self._graph,
            config=self._config.links,
        )

        # Initialize embedding model if configured
        if self._config.embedding.provider and self._config.embedding.provider != "none":
            try:
                self._embedding = create_embedding_model(self._config.embedding)
                logger.info(f"Embedding model initialized: {self._config.embedding.provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")

        # Initialize vector backend if configured
        if self._config.vector.backend not in ("none", None) and self._embedding:
            try:
                self._vector = create_vector_backend(self._config.vector)
                self._vector.connect()
                logger.info(f"Vector backend initialized: {self._config.vector.backend}")
            except Exception as e:
                logger.warning(f"Failed to initialize vector backend: {e}")

        # Initialize LLM provider if configured
        if self._config.llm.orchestrator and self._config.llm.orchestrator != "none":
            try:
                from akm.query.llm.langchain import LangChainLLMProvider
                self._llm = LangChainLLMProvider(config=self._config.llm.langchain)
                logger.info(f"LLM provider initialized: {self._config.llm.langchain.provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider: {e}")

        # Initialize query engine if all dependencies available
        if self._graph and self._llm:
            self._query_engine = QueryEngine(
                graph=self._graph,
                vector=self._vector,
                llm_provider=self._llm,
                link_manager=self._link_manager,
                embedding_model=self._embedding,
                config=self._config,
            )
            logger.info("Query engine initialized")

        # Ensure data directory exists
        self._config.ensure_data_dir()

        self._initialized = True
        logger.info(f"AKM initialized for project: {self._config.project_name}")

    def disconnect(self) -> None:
        """Disconnect from all backends."""
        if self._graph:
            self._graph.disconnect()
        if self._vector:
            self._vector.disconnect()
        self._initialized = False
        logger.info("AKM disconnected")

    # =========================================================================
    # Entity Operations
    # =========================================================================

    def add_entity(
        self,
        name: str,
        entity_type: str = "generic",
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Entity:
        """
        Add a new entity to the knowledge graph.

        Args:
            name: Entity name
            entity_type: Type of entity
            description: Optional description
            properties: Additional properties

        Returns:
            The created Entity
        """
        entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
            **kwargs,
        )

        # Apply domain transformation if configured
        if self._domain_transformer:
            mapped = self._domain_transformer.map_entity(entity)
            if mapped:
                entity = mapped

        created = self.graph.create_entity(entity)
        logger.debug(f"Created entity: {created.name} ({created.id})")
        return created

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.graph.get_entity(entity_id)

    def update_entity(self, entity: Entity) -> Entity:
        """Update an existing entity."""
        return self.graph.update_entity(entity)

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID."""
        return self.graph.delete_entity(entity_id)

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Find entities matching criteria."""
        return self.graph.find_entities(entity_type, properties, limit)

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = "RELATED_TO",
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Relationship:
        """
        Add a relationship between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            properties: Additional properties

        Returns:
            The created Relationship
        """
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties or {},
            **kwargs,
        )

        # Apply domain transformation
        if self._domain_transformer:
            mapped = self._domain_transformer.map_relationship(relationship)
            if mapped:
                relationship = mapped

        created = self.graph.create_relationship(source_id, target_id, relationship)
        logger.debug(f"Created relationship: {relationship_type} ({source_id} -> {target_id})")
        return created

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        return self.graph.get_relationships(entity_id, direction, relationship_type)

    # =========================================================================
    # Link Operations (Adaptive Links)
    # =========================================================================

    def create_soft_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "inferred",
        pattern_source: Optional[str] = None,
        pattern_confidence: float = 0.5,
        semantic_similarity: Optional[float] = None,
        **kwargs: Any,
    ) -> Link:
        """
        Create a soft (unvalidated) link between entities.

        Soft links are automatically created when patterns are detected.
        They can be strengthened through validation or weakened through decay.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            link_type: Type of link
            pattern_source: What detected this pattern
            pattern_confidence: Confidence in the pattern (0-1)
            semantic_similarity: Optional semantic similarity score

        Returns:
            The created Link
        """
        return self.link_manager.create_soft_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            pattern_source=pattern_source,
            pattern_confidence=pattern_confidence,
            semantic_similarity=semantic_similarity,
            **kwargs,
        )

    def validate_link(
        self,
        link_id: str,
        is_positive: bool,
        strength: Optional[float] = None,
    ) -> Link:
        """
        Validate a link (user interaction).

        When a user finds a link helpful (or not), call this method to
        strengthen or weaken the link.

        Args:
            link_id: Link ID
            is_positive: Whether validation is positive
            strength: Validation strength (0-1)

        Returns:
            Updated Link
        """
        return self.link_manager.validate_link(link_id, is_positive, strength)

    def get_links(
        self,
        entity_id: str,
        min_weight: float = 0.0,
        status: Optional[LinkStatus] = None,
    ) -> List[Link]:
        """Get links for an entity."""
        return self.link_manager.get_links(entity_id, min_weight, status)

    def get_link(self, link_id: str) -> Optional[Link]:
        """Get a link by ID."""
        return self.link_manager.get_link(link_id)

    def run_link_decay(self) -> int:
        """
        Run time-based decay on all links.

        Links that haven't been validated or used will decay over time.
        Links that fall below the demotion threshold are archived.

        Returns:
            Number of links that were decayed
        """
        return self.link_manager.run_decay()

    def get_link_stats(self) -> Dict[str, Any]:
        """Get statistics about links."""
        return self.link_manager.get_stats()

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    def traverse(
        self,
        start_id: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        include_links: bool = True,
        min_link_weight: float = 0.3,
    ) -> TraversalResult:
        """
        Traverse the graph from a starting entity.

        Args:
            start_id: Starting entity ID
            depth: Traversal depth
            relationship_types: Filter by relationship types
            include_links: Include adaptive links in result
            min_link_weight: Minimum link weight to include

        Returns:
            TraversalResult with entities, relationships, and links
        """
        result = self.graph.traverse(start_id, depth, relationship_types)

        if include_links:
            entity_ids = [str(e.id) for e in result.entities]
            entity_ids.append(start_id)
            result.links = self.link_manager.get_links_in_subgraph(
                entity_ids, min_link_weight
            )

        return result

    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """Get neighboring entities."""
        return self.graph.get_neighbors(entity_id, depth, relationship_types)

    # =========================================================================
    # Search Operations (placeholder - requires vector backend)
    # =========================================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Note: Requires vector backend to be initialized.

        Args:
            query: Search query
            top_k: Number of results
            search_type: "semantic", "keyword", or "hybrid"
            filters: Optional metadata filters

        Returns:
            List of SearchResult
        """
        if not self._vector:
            logger.warning("Vector backend not initialized. Search not available.")
            return []

        if not self._embedding:
            logger.warning("Embedding model not initialized. Search not available.")
            return []

        # Generate query embedding
        query_embedding = self._embedding.embed_text(query)

        # Perform search based on type
        if search_type == "hybrid":
            results = self._vector.hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        else:
            # Semantic search only
            results = self._vector.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )

        return results

    # =========================================================================
    # Query Operations (placeholder - requires query engine)
    # =========================================================================

    def query(
        self,
        question: str,
        context_entities: Optional[List[str]] = None,
        max_hops: int = 2,
        include_reasoning: bool = True,
    ) -> QueryResult:
        """
        Execute a natural language query.

        Note: Requires query engine to be initialized.

        Args:
            question: Natural language question
            context_entities: Optional entity IDs for context
            max_hops: Maximum graph traversal hops
            include_reasoning: Include reasoning path in result

        Returns:
            QueryResult with answer and sources
        """
        if not self._query_engine:
            logger.warning("Query engine not initialized. Configure LLM provider.")
            return QueryResult(
                answer="Query engine not initialized. Please configure LLM provider.",
                sources=[],
                entities_involved=[],
                relationships_involved=[],
                links_involved=[],
                reasoning_path=[],
                confidence=0.0,
            )

        return self._query_engine.query(
            question=question,
            context_entities=context_entities,
            max_hops=max_hops,
            include_reasoning=include_reasoning,
        )

    # =========================================================================
    # Domain Operations
    # =========================================================================

    def set_domain(self, transformer: DomainTransformer) -> None:
        """Set the domain transformer."""
        self._domain_transformer = transformer
        logger.info(f"Domain set to: {transformer.domain_name}")

    def load_domain(self, domain_name: str) -> None:
        """
        Load a built-in domain adapter.

        Args:
            domain_name: Name of the domain (e.g., "software_engineering")

        Raises:
            ConfigurationError: If domain adapter not found
        """
        transformer = get_domain_transformer(domain_name)
        if transformer is None:
            raise ConfigurationError(f"Domain adapter not found: {domain_name}")

        self._domain_transformer = transformer
        logger.info(f"Domain loaded: {transformer.domain_name}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @contextmanager
    def session(self) -> Generator["AKM", None, None]:
        """Context manager for sessions."""
        self.connect()
        try:
            yield self
        finally:
            self.disconnect()

    def __enter__(self) -> "AKM":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """Execute a native graph query (e.g., Cypher)."""
        return self.graph.execute_query(query, parameters)
