# Adaptive Knowledge Mesh (AKM) Framework - Implementation Plan

## Overview

Build an open-source Python framework that combines knowledge graphs, vector databases, and graph neural networks into an adaptive, self-organizing knowledge system with domain-agnostic core and pluggable domain adapters.

## Technology Choices (Based on User Selections)

| Component | Choice |
|-----------|--------|
| Language | Python 3.10+ |
| Knowledge Graph | Neo4j (primary) |
| Vector DB | ChromaDB (primary) |
| LLM Orchestration | LangChain/LlamaIndex |
| Domain Transformer | Hybrid (schema + fine-tuning + prompts) |
| Link Decay | Time-based exponential decay |
| Deployment | Library/SDK first (pip installable) |
| License | Apache 2.0 |

---

## Package Structure

```
akm/
├── core/           # Data models, interfaces, config, exceptions
├── graph/          # Knowledge graph layer (Neo4j, memory backends)
├── vector/         # Vector DB layer (ChromaDB, embeddings, search)
├── links/          # Adaptive link lifecycle (soft links, decay, validation)
├── gnn/            # Graph Neural Networks (link prediction, community detection)
├── domain/         # Domain transformer middleware (schema, embeddings, prompts)
├── query/          # Query engine / LLM reasoner
├── ingestion/      # Data ingestion pipeline (connectors, extraction)
├── plugins/        # Plugin system
├── adapters/       # Pre-built domain adapters (software_engineering, research)
├── api/            # Programmatic API (AKM client class)
└── cli/            # Command-line interface
```

---

## Implementation Phases

### Phase 1: Core Foundation (Week 1-2)
**Files to create:**
- `akm/__init__.py` - Package init with version
- `akm/core/models.py` - Entity, Relationship, Link, Document, SearchResult
- `akm/core/interfaces.py` - GraphBackend, VectorBackend, DomainTransformer, LLMProvider protocols
- `akm/core/config.py` - Pydantic configuration with YAML loading
- `akm/core/exceptions.py` - Custom exception hierarchy
- `pyproject.toml` - Package metadata, dependencies

**Key Models:**
- `Entity` - Graph node with embeddings, confidence, domain_type
- `Relationship` - Graph edge with properties
- `Link` - Adaptive link with `LinkWeight` (value, decay_rate, validation tracking)
- `LinkStatus` enum: SOFT -> VALIDATING -> VALIDATED -> DECAYING -> ARCHIVED

### Phase 2: Graph Layer (Week 2-3)
**Files to create:**
- `akm/graph/base.py` - Abstract GraphBackend
- `akm/graph/neo4j/client.py` - Neo4j connection management
- `akm/graph/neo4j/repository.py` - CRUD operations
- `akm/graph/neo4j/query_builder.py` - Cypher query abstraction
- `akm/graph/memory/client.py` - In-memory backend for testing

### Phase 3: Vector Layer (Week 3-4)
**Files to create:**
- `akm/vector/base.py` - Abstract VectorBackend
- `akm/vector/chromadb/client.py` - ChromaDB integration
- `akm/vector/chromadb/chunking.py` - Document chunking strategies
- `akm/vector/embeddings/sentence_transformers.py` - Default embedding model
- `akm/vector/search/hybrid.py` - Semantic + keyword hybrid search

### Phase 4: Adaptive Link Lifecycle (Week 4-5)
**Files to create:**
- `akm/links/manager.py` - LinkManager orchestration
- `akm/links/soft_link.py` - Pattern-based soft link creation
- `akm/links/decay.py` - Exponential time-based decay: `weight * exp(-decay_rate * hours)`
- `akm/links/validation.py` - User interaction strengthens/weakens links

**Decay Algorithm:**
```python
def apply_decay(weight: float, decay_rate: float, hours: float) -> float:
    return max(0.0, weight * math.exp(-decay_rate * hours))
```

### Phase 5: Query Engine (Week 5-6)
**Files to create:**
- `akm/query/engine.py` - QueryEngine orchestration
- `akm/query/traversal.py` - Graph traversal with semantic context
- `akm/query/temporal.py` - Decision lineage queries
- `akm/query/llm/langchain.py` - LangChain integration
- `akm/query/synthesis.py` - Result synthesis with reasoning paths

### Phase 6: Domain Transformer (Week 6-7)
**Files to create:**
- `akm/domain/transformer.py` - Base DomainTransformer class
- `akm/domain/schema/loader.py` - YAML schema loader
- `akm/domain/schema/mapper.py` - Generic-to-domain entity mapping
- `akm/domain/prompts/context.py` - Domain context injection
- `akm/adapters/software_engineering/` - Example domain adapter

**Domain Schema Format (YAML):**
```yaml
domain_name: "software_engineering"
entity_types:
  - name: "Repository"
    base_type: "document"
    properties: {url: string, language: string}
relationship_types:
  - name: "IMPORTS"
    source_types: ["CodeFile"]
    target_types: ["CodeFile", "Dependency"]
generic_to_domain_entity_map:
  document: "CodeFile"
  person: "Developer"
```

### Phase 7: Ingestion Pipeline (Week 7-8)
**Files to create:**
- `akm/ingestion/pipeline.py` - IngestionPipeline orchestration
- `akm/ingestion/connectors/files.py` - File system connector
- `akm/ingestion/connectors/base.py` - Connector interface
- `akm/ingestion/extraction/ner.py` - Named entity recognition
- `akm/ingestion/extraction/relations.py` - Relationship extraction

### Phase 8: GNN Layer (Week 8-9)
**Files to create:**
- `akm/gnn/base.py` - GNNManager
- `akm/gnn/link_prediction.py` - Link prediction with GraphSAGE/PyTorch Geometric
- `akm/gnn/community_detection.py` - Louvain community detection

### Phase 9: API & CLI (Week 9-10)
**Files to create:**
- `akm/api/client.py` - Main `AKM` class with fluent API
- `akm/cli/main.py` - Click-based CLI
- `akm/cli/commands/` - init, ingest, query, search, train, serve

**API Usage Example:**
```python
from akm import AKM

akm = AKM.from_config("./akm.yaml")
akm.ingest("./documents/")
result = akm.query("What are the main architectural patterns?")
```

**CLI Usage:**
```bash
akm init --name my_project --domain software_engineering
akm ingest ./data --recursive
akm query "Who worked on the authentication module?"
akm train --model link_prediction --epochs 100
```

---

## Key Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "neo4j>=5.0",
    "chromadb>=0.4",
    "sentence-transformers>=2.2",
    "langchain>=0.1",
    "langchain-openai>=0.0.5",
    "torch>=2.0",
    "torch-geometric>=2.3",  # Optional
    "click>=8.0",
    "pyyaml>=6.0",
    "rich>=13.0",  # For CLI output
]
```

---

## Verification Plan

### 1. Unit Tests
```bash
pytest tests/unit/ -v
```
- Test data models serialization/validation
- Test decay algorithm mathematics
- Test link weight validation logic
- Test schema loading

### 2. Integration Tests
```bash
# Start Neo4j (Docker)
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

pytest tests/integration/ -v
```
- Test Neo4j CRUD operations
- Test ChromaDB embedding storage/retrieval
- Test full ingestion pipeline
- Test query engine with graph + vector

### 3. End-to-End Test
```bash
# Initialize project
akm init --name test_project --domain software_engineering
cd test_project

# Ingest sample data
akm ingest ./sample_docs/

# Query
akm query "What are the main components?"

# Verify link decay
akm decay
```

### 4. Manual Verification
1. Create entities and relationships via API
2. Verify soft links are created from co-occurrence patterns
3. Simulate user validation - verify link weights increase
4. Wait/simulate time passage - verify decay reduces weights
5. Query with temporal context - verify decision lineage works

---

## Critical Files Summary

| File | Purpose |
|------|---------|
| `akm/core/models.py` | All data models including Link with adaptive weights |
| `akm/core/interfaces.py` | Abstract protocols for backends |
| `akm/links/manager.py` | Link lifecycle orchestration |
| `akm/links/decay.py` | Time-based decay implementation |
| `akm/domain/transformer.py` | Generic-to-domain mapping |
| `akm/query/engine.py` | LLM-powered query orchestration |
| `akm/api/client.py` | Main public API |

---

## Success Criteria

1. Can pip install the package
2. Can initialize a project with CLI
3. Can ingest documents and extract entities/relationships
4. Soft links are automatically created from patterns
5. Link weights decay over time without activity
6. User validation strengthens relevant links
7. Queries return context-aware answers with reasoning paths
8. Domain adapters can customize entity/relationship types
9. All tests pass (unit, integration, e2e)
