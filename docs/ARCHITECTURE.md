# AKM Architecture Guide

This document provides a comprehensive overview of the Adaptive Knowledge Mesh (AKM) framework architecture.

## System Overview

AKM is built on a layered architecture that separates concerns and allows for pluggable backends:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Application Layer                           │
│                    (CLI / Python API / REST API)                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                         API Client Layer                            │
│                        akm/api/client.py                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Entity    │  │Relationship │  │    Link     │  │   Query    │ │
│  │ Operations  │  │ Operations  │  │  Operations │  │ Operations │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                     Domain Transformer Layer                        │
│                     akm/domain/transformer.py                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Generic Entity/Relationship → Domain-Specific Mapping       │  │
│  │  Schema Loading │ Property Validation │ Prompt Injection     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼───────┐         ┌───────▼───────┐         ┌───────▼───────┐
│  Graph Layer  │         │ Vector Layer  │         │  Links Layer  │
│  akm/graph/   │         │  akm/vector/  │         │  akm/links/   │
├───────────────┤         ├───────────────┤         ├───────────────┤
│ • Neo4j       │         │ • ChromaDB    │         │ • LinkManager │
│ • In-Memory   │         │ • Embeddings  │         │ • Soft Links  │
│ • NetworkX    │         │ • Hybrid      │         │ • Decay       │
│               │         │   Search      │         │ • Validation  │
└───────────────┘         └───────────────┘         └───────────────┘
        │                         │                         │
┌───────▼─────────────────────────▼─────────────────────────▼───────┐
│                          Core Layer                                │
│                         akm/core/                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   Models    │  │ Interfaces  │  │   Config    │  │Exceptions │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Core Layer (`akm/core/`)

The foundation layer containing data models, interfaces, and configuration.

#### Models (`models.py`)

```python
# Key data models
class Entity(AKMBaseModel):
    """Node in the knowledge graph."""
    name: str
    entity_type: Union[EntityType, str]
    description: Optional[str]
    properties: Dict[str, Any]
    embedding: Optional[List[float]]
    confidence: float
    domain_type: Optional[str]  # Domain-specific type override

class Relationship(AKMBaseModel):
    """Edge in the knowledge graph."""
    source_id: UUID
    target_id: UUID
    relationship_type: Union[RelationshipType, str]
    properties: Dict[str, Any]
    confidence: float
    domain_type: Optional[str]

class Link(AKMBaseModel):
    """Adaptive link with evolving weight."""
    source_id: UUID
    target_id: UUID
    status: LinkStatus  # SOFT → VALIDATING → VALIDATED → DECAYING → ARCHIVED
    weight: LinkWeight
    pattern_source: Optional[str]
    semantic_similarity: Optional[float]
    co_occurrence_count: int

class LinkWeight(BaseModel):
    """Weight with decay and validation tracking."""
    value: float  # 0.0 - 1.0
    decay_rate: float  # Exponential decay per hour
    validation_count: int
    positive_validations: int
    negative_validations: int
```

#### Interfaces (`interfaces.py`)

Abstract protocols for pluggable backends:

```python
class GraphBackend(Protocol):
    """Interface for graph storage backends."""
    def connect(self) -> None: ...
    def create_entity(self, entity: Entity) -> Entity: ...
    def get_entity(self, entity_id: UUID) -> Optional[Entity]: ...
    def create_relationship(self, rel: Relationship) -> Relationship: ...
    def traverse(self, start_id: UUID, depth: int) -> TraversalResult: ...

class VectorBackend(Protocol):
    """Interface for vector storage backends."""
    def add_embedding(self, id: str, embedding: List[float], metadata: Dict) -> None: ...
    def search(self, query_embedding: List[float], k: int) -> List[SearchResult]: ...

class DomainTransformer(Protocol):
    """Interface for domain mapping."""
    def map_entity(self, entity: Entity) -> Optional[Entity]: ...
    def map_relationship(self, rel: Relationship) -> Optional[Relationship]: ...
    def get_prompt_context(self, query: str) -> str: ...
```

### 2. Graph Layer (`akm/graph/`)

Handles knowledge graph storage and traversal.

#### Neo4j Backend

```python
from akm.graph.neo4j.client import Neo4jGraphBackend

backend = Neo4jGraphBackend(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password"
)
backend.connect()

# Creates Cypher queries internally
entity = backend.create_entity(Entity(name="MyService", entity_type="component"))
```

#### In-Memory Backend

```python
from akm.graph.memory.client import InMemoryGraphBackend

backend = InMemoryGraphBackend(persist_path="./graph.json")
# Uses NetworkX for graph operations
```

### 3. Vector Layer (`akm/vector/`)

Handles semantic embeddings and similarity search.

#### ChromaDB Integration

```python
from akm.vector.chromadb.client import ChromaDBVectorBackend
from akm.vector.embeddings.sentence_transformers import SentenceTransformerEmbedding

embedding_model = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
vector_backend = ChromaDBVectorBackend(
    persist_directory="./.akm/chroma",
    collection_name="documents"
)

# Embed and store
embedding = embedding_model.embed("UserService handles user authentication")
vector_backend.add_embedding(id="chunk_1", embedding=embedding, metadata={"source": "docs"})

# Semantic search
results = vector_backend.search(query_embedding, k=10)
```

#### Hybrid Search

```python
from akm.vector.search.hybrid import HybridSearch

search = HybridSearch(vector_backend, keyword_weight=0.3, semantic_weight=0.7)
results = search.search("authentication module", k=5)
```

### 4. Links Layer (`akm/links/`)

The unique innovation of AKM - adaptive, evolving relationships.

#### Link Manager

```python
from akm.links.manager import LinkManager

link_manager = LinkManager(config=links_config)

# Create soft link from detected pattern
link = link_manager.create_soft_link(
    source_id=entity1.id,
    target_id=entity2.id,
    pattern_source="co_occurrence",
    initial_weight=0.5
)

# User validates the link
link_manager.validate_link(link.id, is_positive=True)  # Weight increases

# Apply decay to all links
link_manager.run_decay()  # Stale links lose weight

# Get strong/weak links
strong_links = link_manager.get_links_above_threshold(0.8)
weak_links = link_manager.get_links_below_threshold(0.2)
```

#### Pattern Detection

```python
from akm.links.soft_link import PatternDetector, SoftLinkCreator

detector = PatternDetector()
creator = SoftLinkCreator(link_manager)

# Detect patterns in text
patterns = detector.detect_co_occurrence(entities, text, window_size=50)
patterns += detector.detect_semantic_similarity(entities, threshold=0.8)
patterns += detector.detect_explicit_mentions(text)

# Create soft links from patterns
for pattern in patterns:
    creator.create_from_pattern(pattern)
```

#### Decay Algorithm

```python
from akm.links.decay import DecayManager

decay_manager = DecayManager(
    decay_rate=0.01,  # Per hour
    decay_interval_hours=24,
    minimum_weight=0.1
)

# Decay formula: weight * exp(-decay_rate * hours)
# With decay_rate=0.01, half-life is ~69 hours

decay_manager.apply_decay_to_all(links)
```

### 5. Domain Layer (`akm/domain/`)

Maps generic entities to domain-specific types.

```python
from akm.domain.transformer import BaseDomainTransformer

transformer = BaseDomainTransformer(schema_path="./schemas/software_engineering.yaml")

# Map generic entity to domain type
generic_entity = Entity(name="main.py", entity_type="document")
domain_entity = transformer.map_entity(generic_entity)
# domain_entity.domain_type = "CodeFile"

# Get domain context for LLM prompts
context = transformer.get_prompt_context("What does UserService do?")
```

### 6. Query Layer (`akm/query/`)

Orchestrates semantic search, graph traversal, and LLM synthesis.

```python
from akm.query.engine import QueryEngine

engine = QueryEngine(
    graph_backend=graph,
    vector_backend=vector,
    llm_provider=langchain_provider,
    domain_transformer=transformer
)

result = engine.query("What services depend on AuthService?")
# Returns: QueryResult with answer, sources, entities, reasoning_path
```

### 7. Ingestion Layer (`akm/ingestion/`)

Handles document processing and entity extraction.

```python
from akm.ingestion.pipeline import IngestionPipeline
from akm.ingestion.connectors.files import FileSystemConnector

pipeline = IngestionPipeline(
    graph_backend=graph,
    vector_backend=vector,
    domain_transformer=transformer
)

# Ingest from file system
connector = FileSystemConnector(base_path="./docs", patterns=["*.md", "*.py"])
stats = pipeline.ingest(connector)
# stats: {files_processed: 100, entities_created: 250, links_created: 180}
```

## Data Flow

### 1. Ingestion Flow

```
Document Source
      │
      ▼
┌─────────────────┐
│  Connector      │ (FileSystem, Slack, API, etc.)
│  reads raw data │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Entity Extractor│ (LLM or rule-based NER)
│ extracts entities│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Domain Transform│ (generic → domain types)
│ maps types      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ Graph │ │Vector │
│Backend│ │Backend│
└───────┘ └───────┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│ Pattern Detector│ (co-occurrence, semantic, etc.)
│ finds patterns  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Link Manager   │ (creates soft links)
│ manages links   │
└─────────────────┘
```

### 2. Query Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ Domain Context  │ (inject domain knowledge)
│ injection       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Vector │ │ Graph │
│Search │ │Travers│
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Context Assembly│ (merge search + graph)
│ builds context  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Synthesis   │ (generate answer)
│ with reasoning  │
└────────┬────────┘
         │
         ▼
   QueryResult
   (answer, sources, reasoning)
```

### 3. Link Lifecycle Flow

```
Pattern Detection
       │
       ▼
┌──────────────┐
│ SOFT Link    │ (initial weight: 0.5)
│ created      │
└──────┬───────┘
       │
  Auto-validate?
   ┌───┴───┐
   │       │
   ▼       ▼
  Yes      No
   │       │
   ▼       │
VALIDATED  │
   │       │
   └───┬───┘
       │
       ▼
┌──────────────┐
│User Feedback │
│ (+0.1/-0.15) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Time Passes  │
│ (24hr decay) │
└──────┬───────┘
       │
       ▼
 weight < 0.1?
   ┌───┴───┐
   │       │
   ▼       ▼
  Yes      No
   │       │
   ▼       │
ARCHIVED   │
   │       │
   └───────┘
```

## Configuration

### YAML Configuration Schema

```yaml
project_name: string
data_dir: string  # Default: ./.akm

graph:
  backend: "neo4j" | "memory"
  neo4j:
    uri: string
    username: string
    password: string
    database: string
  memory:
    persist_path: string | null

vector:
  backend: "chromadb"
  chromadb:
    persist_directory: string
    collection_name: string

embedding:
  provider: "sentence_transformers" | "openai"
  sentence_transformers:
    model_name: string  # Default: all-MiniLM-L6-v2
  openai:
    model_name: string
    api_key: string  # Or use OPENAI_API_KEY env var

llm:
  orchestrator: "langchain" | "llamaindex"
  langchain:
    provider: "openai" | "azure_openai" | "anthropic"
    model_name: string
    temperature: float

links:
  initial_soft_link_weight: float  # Default: 0.5
  promotion_threshold: float  # Default: 0.8
  demotion_threshold: float  # Default: 0.2
  decay:
    enabled: boolean
    decay_rate: float  # Default: 0.01 (per hour)
    decay_interval_hours: int  # Default: 24
    minimum_weight: float  # Default: 0.1

domain:
  name: string
  schema_path: string  # Path to domain YAML schema
```

## Extension Points

### Custom Graph Backend

```python
from akm.core.interfaces import GraphBackend

class MyCustomGraphBackend(GraphBackend):
    def connect(self) -> None:
        # Connect to your graph database
        pass

    def create_entity(self, entity: Entity) -> Entity:
        # Store entity
        pass

    # Implement other methods...
```

### Custom Domain Transformer

```python
from akm.domain.transformer import BaseDomainTransformer

class MedicalDomainTransformer(BaseDomainTransformer):
    def map_entity(self, entity: Entity) -> Optional[Entity]:
        # Custom mapping logic for medical domain
        if "patient" in entity.name.lower():
            entity.domain_type = "Patient"
        return entity

    def get_extraction_prompt(self) -> str:
        return """Extract medical entities:
        - Patient names
        - Diagnoses (ICD-10 codes)
        - Medications
        - Procedures
        """
```

### Custom Pattern Detector

```python
from akm.links.soft_link import PatternDetector

class CodePatternDetector(PatternDetector):
    def detect_import_patterns(self, code: str) -> List[Pattern]:
        # Detect import statements
        import_regex = r"from\s+(\w+)\s+import\s+(\w+)"
        # Return detected patterns
        pass
```

## Performance Considerations

### Batch Operations

```python
# Batch entity creation
entities = [Entity(name=f"entity_{i}") for i in range(1000)]
akm.add_entities_batch(entities)

# Batch embedding
embeddings = embedding_model.embed_batch(texts)
```

### Connection Pooling

Neo4j backend uses connection pooling by default:

```yaml
graph:
  neo4j:
    max_connection_pool_size: 50
    connection_acquisition_timeout: 60
```

### Decay Optimization

Run decay during off-peak hours:

```python
# Schedule decay to run daily at 2 AM
scheduler.schedule(link_manager.run_decay, cron="0 2 * * *")
```

## Testing Strategy

```
tests/
├── unit/
│   ├── core/test_models.py      # Data model validation
│   ├── links/test_decay.py      # Decay math verification
│   └── domain/test_mapper.py    # Schema mapping logic
├── integration/
│   ├── test_akm_client.py       # Full client operations
│   └── test_neo4j.py            # Neo4j integration
└── e2e/
    └── test_full_workflow.py    # End-to-end ingestion→query
```

Run tests:

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires Neo4j)
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=akm --cov-report=html
```
