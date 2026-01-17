"""Core interfaces and protocols for the AKM framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
)

from akm.core.models import (
    Chunk,
    Document,
    DomainSchema,
    Entity,
    ExtractedEntity,
    ExtractedRelationship,
    GraphData,
    Link,
    NodeEmbeddings,
    Relationship,
    SearchResult,
    TrainingResult,
    TraversalResult,
)

EntityT = TypeVar("EntityT", bound=Entity)
RelationshipT = TypeVar("RelationshipT", bound=Relationship)


class GraphBackend(ABC, Generic[EntityT, RelationshipT]):
    """Abstract interface for knowledge graph backends."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the graph database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the graph database."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        pass

    @abstractmethod
    def create_entity(self, entity: EntityT) -> EntityT:
        """Create a new entity in the graph."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[EntityT]:
        """Retrieve an entity by ID."""
        pass

    @abstractmethod
    def update_entity(self, entity: EntityT) -> EntityT:
        """Update an existing entity."""
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID."""
        pass

    @abstractmethod
    def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[EntityT]:
        """Find entities matching criteria."""
        pass

    @abstractmethod
    def create_relationship(
        self, source_id: str, target_id: str, relationship: RelationshipT
    ) -> RelationshipT:
        """Create a relationship between two entities."""
        pass

    @abstractmethod
    def get_relationship(self, relationship_id: str) -> Optional[RelationshipT]:
        """Retrieve a relationship by ID."""
        pass

    @abstractmethod
    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
    ) -> List[RelationshipT]:
        """Get relationships for an entity."""
        pass

    @abstractmethod
    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship by ID."""
        pass

    @abstractmethod
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """Execute a native query (e.g., Cypher for Neo4j)."""
        pass

    @abstractmethod
    def traverse(
        self,
        start_id: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
    ) -> TraversalResult:
        """Traverse the graph from a starting node."""
        pass

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> List[EntityT]:
        """Get neighboring entities."""
        pass


class VectorBackend(ABC):
    """Abstract interface for vector database backends."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the vector database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the vector database."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        pass

    @abstractmethod
    def create_collection(
        self, name: str, dimension: int, metadata: Optional[Dict] = None
    ) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        pass

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        pass

    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        documents: List[Chunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Add documents to a collection."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Perform semantic search."""
        pass

    @abstractmethod
    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Perform hybrid semantic + keyword search."""
        pass

    @abstractmethod
    def delete_documents(self, collection_name: str, document_ids: List[str]) -> bool:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def get_document(
        self, collection_name: str, document_id: str
    ) -> Optional[SearchResult]:
        """Get a document by ID."""
        pass


class EmbeddingModel(ABC):
    """Abstract interface for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed."""
        pass

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Async version of generate."""
        pass

    @abstractmethod
    def generate_with_context(
        self,
        prompt: str,
        context: List[SearchResult],
        **kwargs: Any,
    ) -> str:
        """Generate with additional context items."""
        pass

    @abstractmethod
    def extract_entities(
        self, text: str, entity_types: Optional[List[str]] = None
    ) -> List[ExtractedEntity]:
        """Extract entities from text using LLM."""
        pass

    @abstractmethod
    def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        relationship_types: Optional[List[str]] = None,
    ) -> List[ExtractedRelationship]:
        """Extract relationships from text using LLM."""
        pass


class DataConnector(ABC):
    """Abstract interface for data source connectors."""

    @property
    @abstractmethod
    def connector_type(self) -> str:
        """Return the connector type identifier."""
        pass

    @abstractmethod
    def connect(self, **config: Any) -> None:
        """Connect to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass

    @abstractmethod
    def fetch(
        self,
        since: Optional[datetime] = None,
        filters: Optional[Dict] = None,
    ) -> Iterator[Document]:
        """Fetch documents from the source."""
        pass

    @abstractmethod
    async def afetch(
        self,
        since: Optional[datetime] = None,
        filters: Optional[Dict] = None,
    ) -> AsyncIterator[Document]:
        """Async version of fetch."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source."""
        pass


class GNNModel(ABC):
    """Abstract interface for Graph Neural Network models."""

    @abstractmethod
    def train(
        self, graph_data: GraphData, epochs: int = 100, **kwargs: Any
    ) -> TrainingResult:
        """Train the GNN model."""
        pass

    @abstractmethod
    def predict(self, graph_data: GraphData, **kwargs: Any) -> Any:
        """Make predictions using the trained model."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass

    @abstractmethod
    def get_embeddings(self, graph_data: GraphData) -> NodeEmbeddings:
        """Get node embeddings from the model."""
        pass


class DomainTransformer(ABC):
    """Abstract interface for domain transformers."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain name."""
        pass

    @property
    @abstractmethod
    def schema(self) -> DomainSchema:
        """Return the domain schema."""
        pass

    @abstractmethod
    def map_entity(self, generic_entity: Entity) -> Optional[Entity]:
        """Map a generic entity to a domain-specific entity."""
        pass

    @abstractmethod
    def map_relationship(
        self, generic_relationship: Relationship
    ) -> Optional[Relationship]:
        """Map a generic relationship to a domain-specific relationship."""
        pass

    @abstractmethod
    def get_prompt_context(self, query: str) -> str:
        """Get domain-specific context for LLM prompts."""
        pass

    @abstractmethod
    def validate_entity(self, entity: Entity) -> bool:
        """Validate an entity against the domain schema."""
        pass

    @abstractmethod
    def get_embedding_model(self) -> Optional[EmbeddingModel]:
        """Get domain-specific embedding model if available."""
        pass


class ChunkingStrategy(ABC):
    """Abstract interface for document chunking strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy name."""
        pass

    @abstractmethod
    def chunk(
        self,
        document: Document,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Chunk]:
        """Chunk a document into smaller pieces."""
        pass


class Plugin(ABC):
    """Abstract interface for AKM plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the plugin version."""
        pass

    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass

    def on_entity_created(self, entity: Entity) -> None:
        """Hook called when an entity is created."""
        pass

    def on_relationship_created(self, relationship: Relationship) -> None:
        """Hook called when a relationship is created."""
        pass

    def on_link_weight_updated(self, link: Link) -> None:
        """Hook called when a link weight is updated."""
        pass

    def on_query_executed(self, query: str, result: Any) -> None:
        """Hook called after a query is executed."""
        pass
