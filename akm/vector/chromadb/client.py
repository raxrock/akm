"""ChromaDB vector backend implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from akm.core.config import ChromaDBConfig
from akm.core.exceptions import ConnectionError, EmbeddingError
from akm.core.interfaces import VectorBackend
from akm.core.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class ChromaDBBackend(VectorBackend):
    """ChromaDB implementation of the vector backend."""

    def __init__(self, config: ChromaDBConfig) -> None:
        """
        Initialize ChromaDB backend.

        Args:
            config: ChromaDB configuration
        """
        self._config = config
        self._client = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection to ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings

            settings = Settings(
                anonymized_telemetry=self._config.anonymized_telemetry,
            )

            if self._config.host and self._config.port:
                # Client-server mode
                self._client = chromadb.HttpClient(
                    host=self._config.host,
                    port=self._config.port,
                    settings=settings,
                )
            else:
                # Persistent local mode
                persist_dir = Path(self._config.persist_directory)
                persist_dir.mkdir(parents=True, exist_ok=True)

                self._client = chromadb.PersistentClient(
                    path=str(persist_dir),
                    settings=settings,
                )

            self._connected = True
            logger.info(f"Connected to ChromaDB at {self._config.persist_directory}")
        except ImportError:
            raise ConnectionError(
                "chromadb not installed. Install with: pip install chromadb",
                backend="chromadb",
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to ChromaDB: {e}",
                backend="chromadb",
            )

    def disconnect(self) -> None:
        """Close connection to ChromaDB."""
        self._client = None
        self._connected = False
        logger.info("Disconnected from ChromaDB")

    def is_connected(self) -> bool:
        """Check if connected to ChromaDB."""
        return self._connected and self._client is not None

    def _ensure_connected(self) -> None:
        """Ensure we're connected to ChromaDB."""
        if not self.is_connected():
            raise ConnectionError("Not connected to ChromaDB", backend="chromadb")

    def create_collection(
        self,
        name: str,
        dimension: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Create a new collection.

        Args:
            name: Collection name
            dimension: Embedding dimension
            metadata: Optional collection metadata
        """
        self._ensure_connected()

        try:
            collection_metadata = metadata or {}
            collection_metadata["dimension"] = dimension

            self._client.get_or_create_collection(
                name=name,
                metadata=collection_metadata,
            )
            logger.info(f"Created collection: {name}")
        except Exception as e:
            raise EmbeddingError(f"Failed to create collection: {e}")

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        self._ensure_connected()

        try:
            self._client.delete_collection(name=name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection {name}: {e}")
            return False

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        self._ensure_connected()

        try:
            collections = self._client.list_collections()
            return any(c.name == name for c in collections)
        except Exception:
            return False

    def add_documents(
        self,
        collection_name: str,
        documents: List[Chunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """
        Add documents to a collection.

        Args:
            collection_name: Target collection name
            documents: List of document chunks
            embeddings: Corresponding embeddings

        Returns:
            List of document IDs
        """
        self._ensure_connected()

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        try:
            collection = self._client.get_collection(name=collection_name)

            ids = [str(doc.id) for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = []
            for doc in documents:
                # Filter out keys starting with underscore (not allowed by ChromaDB)
                filtered_metadata = {
                    k: v for k, v in doc.metadata.items()
                    if not k.startswith("_") and v is not None
                }
                metadatas.append({
                    "document_id": str(doc.document_id),
                    "chunk_index": doc.chunk_index,
                    "start_char": doc.start_char,
                    "end_char": doc.end_char,
                    **filtered_metadata,
                })

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

            logger.debug(f"Added {len(documents)} documents to {collection_name}")
            return ids
        except Exception as e:
            raise EmbeddingError(f"Failed to add documents: {e}")

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            collection_name: Collection to search
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        self._ensure_connected()

        try:
            collection = self._client.get_collection(name=collection_name)

            where_filter = self._build_where_filter(filters) if filters else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    score = 1.0 / (1.0 + distance)

                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    content = (
                        results["documents"][0][i] if results["documents"] else ""
                    )

                    search_results.append(
                        SearchResult(
                            document_id=metadata.get("document_id", doc_id),
                            chunk_id=doc_id,
                            content=content,
                            score=score,
                            metadata=metadata,
                        )
                    )

            return search_results
        except Exception as e:
            raise EmbeddingError(f"Failed to perform search: {e}")

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid semantic + keyword search.

        Note: ChromaDB doesn't have native hybrid search, so we simulate it
        by combining semantic search with keyword filtering.

        Args:
            collection_name: Collection to search
            query_text: Query text for keyword matching
            query_embedding: Query embedding for semantic matching
            top_k: Number of results
            alpha: Balance between semantic (1.0) and keyword (0.0)
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        self._ensure_connected()

        # For now, perform semantic search with text-based re-ranking
        # A full implementation would use a separate keyword index
        semantic_results = self.search(
            collection_name,
            query_embedding,
            top_k=top_k * 2,  # Fetch more to allow re-ranking
            filters=filters,
        )

        # Simple keyword boost
        query_terms = set(query_text.lower().split())
        for result in semantic_results:
            content_terms = set(result.content.lower().split())
            keyword_overlap = len(query_terms & content_terms) / max(
                len(query_terms), 1
            )
            # Combine scores
            result.score = alpha * result.score + (1 - alpha) * keyword_overlap

        # Re-sort and limit
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:top_k]

    def delete_documents(
        self,
        collection_name: str,
        document_ids: List[str],
    ) -> bool:
        """Delete documents by ID."""
        self._ensure_connected()

        try:
            collection = self._client.get_collection(name=collection_name)
            collection.delete(ids=document_ids)
            logger.debug(f"Deleted {len(document_ids)} documents from {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete documents: {e}")
            return False

    def get_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> Optional[SearchResult]:
        """Get a document by ID."""
        self._ensure_connected()

        try:
            collection = self._client.get_collection(name=collection_name)
            results = collection.get(
                ids=[document_id],
                include=["documents", "metadatas"],
            )

            if results["ids"]:
                return SearchResult(
                    document_id=document_id,
                    chunk_id=document_id,
                    content=results["documents"][0] if results["documents"] else "",
                    score=1.0,
                    metadata=results["metadatas"][0] if results["metadatas"] else {},
                )
            return None
        except Exception:
            return None

    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict:
        """Build ChromaDB where filter from dict."""
        if not filters:
            return None

        # Convert simple key-value pairs to ChromaDB filter format
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
