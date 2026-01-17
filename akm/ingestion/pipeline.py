"""Main ingestion pipeline for processing documents into the knowledge graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from akm.core.interfaces import DomainTransformer, GraphBackend, VectorBackend
from akm.core.models import Document, Entity, Relationship
from akm.ingestion.connectors.base import BaseConnector, ConnectorResult
from akm.ingestion.connectors.files import FileSystemConnector
from akm.ingestion.extraction.ner import EntityExtractor, ExtractionResult
from akm.ingestion.extraction.relations import RelationshipExtractor
from akm.links.manager import LinkManager

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""

    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    entities_created: int = 0
    entities_updated: int = 0
    relationships_created: int = 0
    soft_links_created: int = 0
    documents_indexed: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""

    # Connector settings
    recursive: bool = True
    patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    max_file_size_mb: float = 10.0

    # Extraction settings
    use_llm_extraction: bool = False
    extract_entities: bool = True
    extract_relationships: bool = True
    create_soft_links: bool = True

    # Vector settings
    index_documents: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Processing settings
    batch_size: int = 100
    on_progress: Optional[Callable[[int, int], None]] = None


class IngestionPipeline:
    """
    Main pipeline for ingesting data into the AKM knowledge system.

    The pipeline:
    1. Reads data from connectors (file system, APIs, etc.)
    2. Extracts entities and relationships
    3. Creates graph nodes and edges
    4. Creates soft links for co-occurring entities
    5. Indexes documents in vector store

    Example usage:
        pipeline = IngestionPipeline(
            graph=graph_backend,
            vector=vector_backend,
            link_manager=link_manager,
        )

        stats = pipeline.ingest("./documents", recursive=True)
        print(f"Processed {stats.files_processed} files")
    """

    def __init__(
        self,
        graph: GraphBackend,
        vector: Optional[VectorBackend] = None,
        link_manager: Optional[LinkManager] = None,
        domain_transformer: Optional[DomainTransformer] = None,
        llm_provider: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        collection_name: str = "akm_documents",
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            graph: Graph backend for storing entities/relationships
            vector: Optional vector backend for document indexing
            link_manager: Optional link manager for soft links
            domain_transformer: Optional domain transformer
            llm_provider: Optional LLM provider for extraction
            embedding_model: Optional embedding model for vector indexing
            collection_name: Name of the vector collection
        """
        self.graph = graph
        self.vector = vector
        self.link_manager = link_manager
        self.domain_transformer = domain_transformer
        self.llm_provider = llm_provider
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Initialize extractors
        self.entity_extractor = EntityExtractor(
            use_llm=llm_provider is not None,
            llm_provider=llm_provider,
            domain_transformer=domain_transformer,
        )
        self.relationship_extractor = RelationshipExtractor(
            use_llm=llm_provider is not None,
            llm_provider=llm_provider,
        )

        # Default connector
        self._file_connector: Optional[FileSystemConnector] = None

    def ingest(
        self,
        source: Union[str, Path, BaseConnector],
        config: Optional[IngestionConfig] = None,
    ) -> IngestionStats:
        """
        Ingest data from a source.

        Args:
            source: Path to ingest from, or a connector instance
            config: Optional ingestion configuration

        Returns:
            IngestionStats with processing results
        """
        config = config or IngestionConfig()
        stats = IngestionStats()

        # Get connector
        if isinstance(source, BaseConnector):
            connector = source
        else:
            connector = self._get_file_connector(config)
            source = str(source)

        # Connect
        connector.connect()

        try:
            # Scan and process files
            results = connector.scan(
                source,
                recursive=config.recursive,
                patterns=config.patterns,
            )

            batch: List[ConnectorResult] = []

            for result in results:
                batch.append(result)

                if len(batch) >= config.batch_size:
                    self._process_batch(batch, config, stats)
                    batch = []

                    # Progress callback
                    if config.on_progress:
                        config.on_progress(
                            stats.files_processed,
                            stats.files_processed + stats.files_skipped,
                        )

            # Process remaining batch
            if batch:
                self._process_batch(batch, config, stats)

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            stats.errors.append(str(e))
        finally:
            connector.disconnect()

        logger.info(
            f"Ingestion complete: {stats.files_processed} files, "
            f"{stats.entities_created} entities, "
            f"{stats.relationships_created} relationships"
        )

        return stats

    def ingest_document(
        self,
        content: str,
        source_path: str,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[IngestionConfig] = None,
    ) -> IngestionStats:
        """
        Ingest a single document directly.

        Args:
            content: Document content
            source_path: Source identifier
            content_type: MIME type
            metadata: Optional metadata
            config: Optional configuration

        Returns:
            IngestionStats
        """
        config = config or IngestionConfig()
        stats = IngestionStats()

        result = ConnectorResult(
            source_path=source_path,
            content=content,
            content_type=content_type,
            metadata=metadata or {},
        )

        self._process_single(result, config, stats)
        return stats

    def _get_file_connector(self, config: IngestionConfig) -> FileSystemConnector:
        """Get or create file system connector."""
        if self._file_connector is None:
            self._file_connector = FileSystemConnector(
                exclude_patterns=config.exclude_patterns,
                max_file_size_mb=config.max_file_size_mb,
            )
        return self._file_connector

    def _process_batch(
        self,
        batch: List[ConnectorResult],
        config: IngestionConfig,
        stats: IngestionStats,
    ) -> None:
        """Process a batch of connector results."""
        for result in batch:
            try:
                self._process_single(result, config, stats)
                stats.files_processed += 1
            except Exception as e:
                logger.warning(f"Failed to process {result.source_path}: {e}")
                stats.files_failed += 1
                stats.errors.append(f"{result.source_path}: {e}")

    def _process_single(
        self,
        result: ConnectorResult,
        config: IngestionConfig,
        stats: IngestionStats,
    ) -> None:
        """Process a single connector result."""
        logger.debug(f"Processing: {result.source_path}")

        entities: List[Entity] = []
        relationships: List[Relationship] = []

        # Extract entities
        if config.extract_entities:
            extraction = self.entity_extractor.extract(
                content=result.content,
                content_type=result.content_type,
                source_path=result.source_path,
            )
            entities = extraction.entities

            # Create file/document entity
            file_entity = self._create_file_entity(result)
            entities.insert(0, file_entity)

        # Extract relationships
        if config.extract_relationships and entities:
            rel_result = self.relationship_extractor.extract(
                content=result.content,
                entities=entities,
                language=self._get_language(result.content_type),
                source_path=result.source_path,
            )
            relationships = rel_result.relationships

        # Apply domain transformation
        if self.domain_transformer:
            entities = [
                self.domain_transformer.map_entity(e) or e for e in entities
            ]
            relationships = [
                self.domain_transformer.map_relationship(r) or r for r in relationships
            ]

        # Store entities in graph
        entity_id_map: Dict[str, str] = {}
        for entity in entities:
            try:
                created = self.graph.create_entity(entity)
                entity_id_map[str(entity.id)] = str(created.id)
                stats.entities_created += 1
            except Exception as e:
                logger.warning(f"Failed to create entity {entity.name}: {e}")

        # Store relationships in graph (only if both entities exist)
        for rel in relationships:
            try:
                # Map IDs - skip if entities weren't created
                source_id = entity_id_map.get(rel.source_id)
                target_id = entity_id_map.get(rel.target_id)

                if not source_id or not target_id:
                    logger.debug(
                        f"Skipping relationship: missing entity "
                        f"(source={rel.source_id in entity_id_map}, "
                        f"target={rel.target_id in entity_id_map})"
                    )
                    continue

                self.graph.create_relationship(source_id, target_id, rel)
                stats.relationships_created += 1
            except Exception as e:
                logger.warning(f"Failed to create relationship: {e}")

        # Create soft links for co-occurrences
        if config.create_soft_links and self.link_manager:
            self._create_soft_links(entities, entity_id_map, stats)

        # Index document in vector store
        if config.index_documents and self.vector:
            self._index_document(result, entities, config, stats)

    def _create_file_entity(self, result: ConnectorResult) -> Entity:
        """Create an entity representing the file."""
        return Entity(
            name=Path(result.source_path).name,
            entity_type="file",
            description=f"File: {result.source_path}",
            properties={
                "source_path": result.source_path,
                "content_type": result.content_type,
                "size_bytes": result.size_bytes,
                "checksum": result.checksum,
                **result.metadata,
            },
        )

    def _create_soft_links(
        self,
        entities: List[Entity],
        entity_id_map: Dict[str, str],
        stats: IngestionStats,
    ) -> None:
        """Create soft links between co-occurring entities."""
        if not self.link_manager:
            return

        # Create links between entities in the same file
        entity_ids = list(entity_id_map.values())

        for i, source_id in enumerate(entity_ids):
            for target_id in entity_ids[i + 1:]:
                try:
                    self.link_manager.create_soft_link(
                        source_id=source_id,
                        target_id=target_id,
                        link_type="co_occurrence",
                        pattern_source="file_co_occurrence",
                        pattern_confidence=0.4,
                    )
                    stats.soft_links_created += 1
                except Exception as e:
                    logger.debug(f"Failed to create soft link: {e}")

    def _index_document(
        self,
        result: ConnectorResult,
        entities: List[Entity],
        config: IngestionConfig,
        stats: IngestionStats,
    ) -> None:
        """Index document in vector store."""
        if not self.vector or not self.embedding_model:
            return

        try:
            from uuid import uuid4
            from akm.core.models import Chunk

            # Create chunks from document content
            chunk_size = config.chunk_size
            content = result.content
            chunks = []
            doc_id = uuid4()

            # Simple chunking by size with overlap
            overlap = chunk_size // 4
            start = 0
            chunk_index = 0

            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_content = content[start:end]

                chunk = Chunk(
                    id=uuid4(),
                    content=chunk_content,
                    document_id=doc_id,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "source": result.source_path,
                        "entities": ", ".join([e.name for e in entities[:5]]),
                    },
                )
                chunks.append(chunk)

                start = end - overlap if end < len(content) else end
                chunk_index += 1

            if not chunks:
                return

            # Generate embeddings for all chunks
            chunk_texts = [c.content for c in chunks]
            embeddings = self.embedding_model.embed(chunk_texts)

            # Ensure collection exists
            if not self.vector.collection_exists(self.collection_name):
                dimension = self.embedding_model.dimension
                self.vector.create_collection(self.collection_name, dimension=dimension)

            # Add to vector store
            self.vector.add_documents(self.collection_name, chunks, embeddings)
            stats.documents_indexed += 1
            logger.debug(f"Indexed {len(chunks)} chunks from {result.source_path}")
        except Exception as e:
            logger.warning(f"Failed to index document: {e}")

    def _get_language(self, content_type: str) -> Optional[str]:
        """Get programming language from content type."""
        type_map = {
            "text/x-python": "python",
            "application/javascript": "javascript",
            "application/typescript": "javascript",
            "text/x-java": "java",
            "text/x-go": "go",
        }
        return type_map.get(content_type)


__all__ = ["IngestionPipeline", "IngestionConfig", "IngestionStats"]
