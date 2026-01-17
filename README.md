# Adaptive Knowledge Mesh (AKM)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An open-source Python framework that combines knowledge graphs, vector databases, and graph neural networks into an **adaptive, self-organizing knowledge system**.

## Key Features

- **Adaptive Links**: Self-organizing relationships with time-based decay and user validation strengthening
- **Knowledge Graph Integration**: Neo4j-powered graph storage with in-memory option for development
- **Vector Database**: ChromaDB for semantic embeddings and hybrid search
- **Domain Transformers**: Map generic entities to domain-specific types via pluggable adapters
- **LLM-Powered Queries**: LangChain integration for context-aware question answering
- **Graph Neural Networks**: Optional link prediction and community detection

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AKM Client (Main API)                    │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
      ┌────▼────┐    ┌───▼────┐    ┌───▼────┐
      │  Graph  │    │ Vector │    │  Link  │
      │ Backend │    │Backend │    │Manager │
      └────┬────┘    └───┬────┘    └───┬────┘
           │              │              │
      ┌────▼──────────────▼──────────────▼────┐
      │       Domain Transformer Layer        │
      │     (Generic → Domain Mapping)        │
      └───────────────────────────────────────┘
```

## Installation

```bash
pip install akm
```

### From Requirements File

```bash
git clone https://github.com/raxrock/akm.git
cd akm
pip install -r requirements.txt
```

### Development Installation

```bash
git clone https://github.com/raxrock/akm.git
cd akm
pip install -r requirements-dev.txt
# Or using pyproject.toml
pip install -e ".[dev]"
```

### With GNN Support

```bash
pip install -r requirements-gnn.txt
# Or using pyproject.toml
pip install -e ".[gnn]"
```

## Quick Start

### 1. Configuration

Create an `akm.yaml` configuration file:

```yaml
project_name: "my_knowledge_base"
data_dir: "./.akm"

graph:
  backend: "memory"  # Use "neo4j" for production

vector:
  backend: "chromadb"
  chromadb:
    persist_directory: "./.akm/chroma"
    collection_name: "documents"

embedding:
  provider: "sentence_transformers"
  sentence_transformers:
    model_name: "all-MiniLM-L6-v2"

llm:
  orchestrator: "langchain"
  langchain:
    provider: "azure_openai"
    model_name: "gpt-4"

links:
  initial_soft_link_weight: 0.5
  decay:
    enabled: true
    decay_rate: 0.01
    decay_interval_hours: 24
```

### 2. Python API

```python
from akm import AKM

# Initialize from config
akm = AKM.from_config("./akm.yaml")
akm.connect()

# Add entities
user_service = akm.add_entity(
    name="UserService",
    entity_type="component",
    properties={"language": "python", "version": "2.1.0"}
)

auth_service = akm.add_entity(
    name="AuthService",
    entity_type="component",
    properties={"language": "python"}
)

# Create relationships
akm.add_relationship(
    source_id=user_service.id,
    target_id=auth_service.id,
    relationship_type="DEPENDS_ON"
)

# Create adaptive soft links (auto-discovered relationships)
link = akm.create_soft_link(
    source_id=user_service.id,
    target_id=auth_service.id,
    pattern_source="co_occurrence"
)

# User validates the link (strengthens weight)
akm.validate_link(link.id, is_positive=True)

# Run decay on stale links
akm.run_link_decay()

# Query the knowledge mesh
result = akm.query("What services does UserService depend on?")
print(result.answer)

# Cleanup
akm.disconnect()
```

### 3. CLI Usage

```bash
# Initialize a new project
akm init --name my_project --domain software_engineering

# Ingest documents
akm ingest ./data --recursive

# Query the knowledge mesh
akm query "What are the main components?"

# Train link prediction model
akm train --model link_prediction --epochs 100
```

## Adaptive Link Lifecycle

The core innovation of AKM is the **adaptive link** system. Links evolve based on:

```
SOFT → VALIDATING → VALIDATED → DECAYING → ARCHIVED
```

### Link Weight Decay

Links decay exponentially over time using the formula:

```python
new_weight = current_weight * exp(-decay_rate * hours_elapsed)
```

- **Default decay rate**: 0.01/hour (~69 hour half-life)
- **Minimum weight**: 0.1 (below this, links are archived)
- **Decay interval**: 24 hours between decay runs

### Pattern Detection

Soft links are created from four pattern types:

1. **Co-occurrence**: Entities appearing together in text
2. **Semantic Similarity**: High cosine similarity between embeddings
3. **Explicit Mentions**: Phrases like "X depends on Y"
4. **Structural Patterns**: Entities in the same document section

### User Validation

```python
# Positive validation: increases weight by 0.1
akm.validate_link(link_id, is_positive=True)

# Negative validation: decreases weight by 0.15
akm.validate_link(link_id, is_positive=False)
```

## Domain Transformer

Map generic entities to domain-specific types using YAML schemas:

```yaml
# domains/software_engineering.yaml
domain_name: "software_engineering"
version: "1.0.0"

entity_types:
  - name: "Repository"
    base_type: "document"
    properties:
      url: string
      language: string
    required_properties: ["url"]

  - name: "Developer"
    base_type: "person"
    properties:
      github_username: string
      team: string

relationship_types:
  - name: "COMMITS_TO"
    source_types: ["Developer"]
    target_types: ["Repository"]

  - name: "IMPORTS"
    source_types: ["CodeFile"]
    target_types: ["CodeFile", "Dependency"]

# Generic-to-domain mappings
generic_to_domain_entity_map:
  document: "CodeFile"
  person: "Developer"

generic_to_domain_relationship_map:
  CREATED_BY: "COMMITS_TO"
  REFERENCES: "IMPORTS"
```

See [DOMAIN_TRANSFORMER.md](./docs/DOMAIN_TRANSFORMER.md) for detailed examples.

## Project Structure

```
akm/
├── core/           # Data models, interfaces, config
├── graph/          # Knowledge graph backends (Neo4j, memory)
├── vector/         # Vector DB layer (ChromaDB, embeddings)
├── links/          # Adaptive link lifecycle management
├── domain/         # Domain transformer middleware
├── query/          # Query engine with LLM integration
├── ingestion/      # Data ingestion pipeline
├── gnn/            # Graph Neural Networks (optional)
├── api/            # Main AKM client class
└── cli/            # Command-line interface
```

## Requirements

- Python 3.10+
- Neo4j 5.0+ (optional, can use in-memory graph)
- ChromaDB 0.4+

## Documentation

- [Architecture Guide](./docs/ARCHITECTURE.md)
- [Domain Transformer Guide](./docs/DOMAIN_TRANSFORMER.md)
- [API Reference](./docs/API.md)
- [Contributing](./CONTRIBUTING.md)

## Author

**Rakshith Kumar Karkala**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/rakshithkumarkarkala/)
[![Email](https://img.shields.io/badge/Email-rakshithkk40%40gmail.com-red)](mailto:rakshithkk40@gmail.com)

## License

Apache 2.0 - See [LICENSE](./LICENSE) for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.
