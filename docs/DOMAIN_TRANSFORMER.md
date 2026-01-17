# Domain Transformer Guide

The Domain Transformer is a core component of AKM that maps generic entities and relationships to domain-specific types. This enables AKM to adapt to any knowledge domain through pluggable schemas.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Generic Entity/Relationship                  │
│                 (person, document, CREATED_BY)                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Domain Transformer                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │ Schema Loader │  │ Schema Mapper │  │ Prompt Injector   │   │
│  └───────────────┘  └───────────────┘  └───────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Domain-Specific Entity/Relationship           │
│                (Developer, CodeFile, COMMITS_TO)                │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Define a Domain Schema

Create a YAML file defining your domain:

```yaml
# schemas/software_engineering.yaml
domain_name: "software_engineering"
version: "1.0.0"
description: "Schema for software engineering knowledge graphs"

entity_types:
  - name: "Repository"
    base_type: "document"
    description: "A code repository (GitHub, GitLab, etc.)"
    properties:
      url: string
      language: string
      stars: integer
    required_properties: ["url"]
    examples: ["akm-framework/akm", "facebook/react"]

  - name: "Developer"
    base_type: "person"
    description: "A software developer or contributor"
    properties:
      github_username: string
      email: string
      team: string
    examples: ["John Doe", "Jane Smith"]

  - name: "CodeFile"
    base_type: "document"
    description: "A source code file"
    properties:
      path: string
      language: string
      lines_of_code: integer
    required_properties: ["path"]

  - name: "Component"
    base_type: "concept"
    description: "A software component or service"
    properties:
      version: string
      status: string

relationship_types:
  - name: "COMMITS_TO"
    description: "Developer commits code to repository"
    source_types: ["Developer"]
    target_types: ["Repository"]

  - name: "IMPORTS"
    description: "Code file imports another file or dependency"
    source_types: ["CodeFile"]
    target_types: ["CodeFile", "Dependency"]

  - name: "DEPENDS_ON"
    description: "Component depends on another component"
    source_types: ["Component"]
    target_types: ["Component"]
    bidirectional: false

# Generic-to-domain mappings
generic_to_domain_entity_map:
  document: "CodeFile"
  person: "Developer"
  concept: "Component"

generic_to_domain_relationship_map:
  CREATED_BY: "COMMITS_TO"
  REFERENCES: "IMPORTS"
  RELATED_TO: "DEPENDS_ON"
```

### 2. Initialize the Transformer

```python
from akm.domain.transformer import BaseDomainTransformer

# Load from YAML file
transformer = BaseDomainTransformer(schema_path="./schemas/software_engineering.yaml")

# Or create programmatically
from akm.core.models import DomainSchema, EntityTypeDefinition

schema = DomainSchema(
    domain_name="software_engineering",
    entity_types=[
        EntityTypeDefinition(
            name="Developer",
            base_type="person",
            properties={"github_username": "string", "team": "string"}
        ),
        EntityTypeDefinition(
            name="CodeFile",
            base_type="document",
            properties={"path": "string", "language": "string"},
            required_properties=["path"]
        )
    ],
    generic_to_domain_entity_map={
        "person": "Developer",
        "document": "CodeFile"
    }
)

transformer = BaseDomainTransformer(schema=schema)
```

### 3. Map Entities

```python
from akm.core.models import Entity, EntityType

# Create a generic entity
generic_entity = Entity(
    name="John Doe",
    entity_type=EntityType.PERSON,
    properties={"email": "john@example.com"}
)

# Map to domain-specific type
domain_entity = transformer.map_entity(generic_entity)

print(domain_entity.entity_type)  # EntityType.PERSON
print(domain_entity.domain_type)  # "Developer"
```

### 4. Map Relationships

```python
from akm.core.models import Relationship, RelationshipType

generic_rel = Relationship(
    source_id=developer_entity.id,
    target_id=repo_entity.id,
    relationship_type=RelationshipType.CREATED_BY
)

domain_rel = transformer.map_relationship(generic_rel)

print(domain_rel.relationship_type)  # RelationshipType.CREATED_BY
print(domain_rel.domain_type)        # "COMMITS_TO"
```

## Domain Schema Reference

### Entity Type Definition

```yaml
entity_types:
  - name: "EntityTypeName"           # Required: unique name
    base_type: "person"              # Base type: person, document, concept, etc.
    description: "Description"       # Optional: human-readable description
    properties:                      # Optional: property definitions
      property_name: type            # Types: string, integer, float, boolean, list, object
    required_properties: ["prop"]    # Optional: properties that must be present
    examples: ["Example 1"]          # Optional: examples for type inference
```

### Relationship Type Definition

```yaml
relationship_types:
  - name: "RELATIONSHIP_NAME"        # Required: unique name (UPPER_SNAKE_CASE)
    description: "Description"       # Optional: human-readable description
    source_types: ["EntityType"]     # Required: valid source entity types
    target_types: ["EntityType"]     # Required: valid target entity types
    properties:                      # Optional: relationship properties
      weight: float
    bidirectional: false             # Optional: whether relationship is bidirectional
```

### Generic-to-Domain Mappings

```yaml
# Entity type mappings
generic_to_domain_entity_map:
  person: "DomainPersonType"
  document: "DomainDocumentType"
  concept: "DomainConceptType"
  organization: "DomainOrgType"

# Relationship type mappings
generic_to_domain_relationship_map:
  CREATED_BY: "DOMAIN_CREATED_BY"
  REFERENCES: "DOMAIN_REFERENCES"
  PART_OF: "DOMAIN_CONTAINS"
```

## Advanced Usage

### Schema Mapper for Batch Operations

```python
from akm.domain.schema.mapper import SchemaMapper, MappingRule

mapper = SchemaMapper(schema=schema)

# Add custom mapping rules with conditions
def is_senior_developer(entity):
    return entity.properties.get("years_experience", 0) > 5

mapper.add_entity_rule(MappingRule(
    source_type="person",
    target_type="SeniorDeveloper",
    conditions=[is_senior_developer],
    priority=10  # Higher priority rules are checked first
))

# Batch mapping
entities = [entity1, entity2, entity3]
results = mapper.map_entities_batch(entities)

for result in results:
    print(f"Original: {result.original.entity_type}")
    print(f"Mapped: {result.mapped.domain_type}")
    print(f"Transformations: {result.transformations}")
```

### Property Transformations

```python
from akm.domain.schema.mapper import MappingRule

# Transform properties during mapping
def normalize_email(email):
    return email.lower().strip()

def parse_github_url(url):
    # Extract username from GitHub URL
    import re
    match = re.search(r'github\.com/([^/]+)', url)
    return match.group(1) if match else url

rule = MappingRule(
    source_type="person",
    target_type="Developer",
    property_transforms={
        "email": normalize_email,
        "github_url": parse_github_url
    }
)

mapper.add_entity_rule(rule)
```

### Type Inference

```python
from akm.domain.schema.mapper import TypeInferencer

inferencer = TypeInferencer(schema=schema)

# Infer type from name and context
entity_type, confidence = inferencer.infer_entity_type(
    name="UserService",
    context="The UserService component handles authentication"
)

print(entity_type)   # "Component"
print(confidence)    # 0.7

# Infer relationship type
rel_type, confidence = inferencer.infer_relationship_type(
    source_type="Component",
    target_type="Component",
    context="UserService depends on AuthService"
)

print(rel_type)      # "DEPENDS_ON"
print(confidence)    # 0.8
```

### Prompt Context Injection

```python
# Get domain context for LLM prompts
context = transformer.get_prompt_context("What does the UserService do?")

print(context)
# Output:
# Domain: software_engineering
# Schema for software engineering knowledge graphs
# Available entity types: Repository, Developer, CodeFile, Component
# Available relationship types: COMMITS_TO, IMPORTS, DEPENDS_ON

# Get extraction prompt
extraction_prompt = transformer.get_extraction_prompt()

print(extraction_prompt)
# Output:
# Extract entities and relationships from the text.
#
# Entity types to extract: Repository, Developer, CodeFile, Component
#
# For each entity found, provide:
# 1. The entity name/text
# 2. The entity type
# ...
```

### Entity Validation

```python
# Validate entity against schema
entity = Entity(
    name="main.py",
    entity_type="document",
    domain_type="CodeFile",
    properties={"language": "python"}  # Missing required "path" property
)

is_valid = transformer.validate_entity(entity)
print(is_valid)  # False (missing required property "path")

# With SchemaMapper for detailed validation
is_valid, error = mapper.validate_entity(entity)
print(error)  # "Missing required property: path"
```

## Example Domain Schemas

### 1. Research Domain

```yaml
# schemas/research.yaml
domain_name: "research"
version: "1.0.0"
description: "Schema for academic research knowledge graphs"

entity_types:
  - name: "Paper"
    base_type: "document"
    properties:
      title: string
      doi: string
      year: integer
      venue: string
    required_properties: ["title"]
    examples: ["Attention Is All You Need", "BERT: Pre-training"]

  - name: "Author"
    base_type: "person"
    properties:
      affiliation: string
      h_index: integer
      orcid: string

  - name: "Institution"
    base_type: "organization"
    properties:
      country: string
      type: string  # university, company, research_lab

  - name: "Topic"
    base_type: "concept"
    properties:
      field: string
      keywords: list

relationship_types:
  - name: "AUTHORED"
    source_types: ["Author"]
    target_types: ["Paper"]

  - name: "CITES"
    source_types: ["Paper"]
    target_types: ["Paper"]

  - name: "AFFILIATED_WITH"
    source_types: ["Author"]
    target_types: ["Institution"]

  - name: "COVERS_TOPIC"
    source_types: ["Paper"]
    target_types: ["Topic"]

generic_to_domain_entity_map:
  document: "Paper"
  person: "Author"
  organization: "Institution"
  concept: "Topic"

generic_to_domain_relationship_map:
  CREATED_BY: "AUTHORED"
  REFERENCES: "CITES"
  PART_OF: "AFFILIATED_WITH"
```

**Usage:**

```python
transformer = BaseDomainTransformer(schema_path="./schemas/research.yaml")

# Extract from academic text
paper = Entity(
    name="Attention Is All You Need",
    entity_type="document",
    properties={"year": 2017, "venue": "NeurIPS"}
)

domain_paper = transformer.map_entity(paper)
print(domain_paper.domain_type)  # "Paper"
```

### 2. E-Commerce Domain

```yaml
# schemas/ecommerce.yaml
domain_name: "ecommerce"
version: "1.0.0"

entity_types:
  - name: "Product"
    base_type: "concept"
    properties:
      sku: string
      price: float
      category: string
      in_stock: boolean
    required_properties: ["sku", "price"]

  - name: "Customer"
    base_type: "person"
    properties:
      customer_id: string
      tier: string  # bronze, silver, gold, platinum

  - name: "Order"
    base_type: "event"
    properties:
      order_id: string
      total: float
      status: string

  - name: "Review"
    base_type: "document"
    properties:
      rating: integer
      verified_purchase: boolean

relationship_types:
  - name: "PURCHASED"
    source_types: ["Customer"]
    target_types: ["Product"]

  - name: "REVIEWED"
    source_types: ["Customer"]
    target_types: ["Product"]

  - name: "CONTAINS"
    source_types: ["Order"]
    target_types: ["Product"]

  - name: "SIMILAR_TO"
    source_types: ["Product"]
    target_types: ["Product"]
    bidirectional: true

generic_to_domain_entity_map:
  person: "Customer"
  concept: "Product"
  event: "Order"
  document: "Review"
```

**Usage:**

```python
transformer = BaseDomainTransformer(schema_path="./schemas/ecommerce.yaml")

customer = Entity(
    name="John Smith",
    entity_type="person",
    properties={"customer_id": "C12345", "tier": "gold"}
)

domain_customer = transformer.map_entity(customer)
print(domain_customer.domain_type)  # "Customer"

# Validate
print(transformer.validate_entity(domain_customer))  # True
```

### 3. Healthcare Domain

```yaml
# schemas/healthcare.yaml
domain_name: "healthcare"
version: "1.0.0"

entity_types:
  - name: "Patient"
    base_type: "person"
    properties:
      patient_id: string
      date_of_birth: string
      blood_type: string
    required_properties: ["patient_id"]

  - name: "Diagnosis"
    base_type: "concept"
    properties:
      icd_code: string
      description: string
      severity: string

  - name: "Medication"
    base_type: "concept"
    properties:
      ndc_code: string
      dosage: string
      frequency: string

  - name: "Provider"
    base_type: "person"
    properties:
      npi: string
      specialty: string

  - name: "Encounter"
    base_type: "event"
    properties:
      encounter_id: string
      encounter_type: string
      date: string

relationship_types:
  - name: "HAS_DIAGNOSIS"
    source_types: ["Patient"]
    target_types: ["Diagnosis"]

  - name: "PRESCRIBED"
    source_types: ["Provider"]
    target_types: ["Medication"]

  - name: "TAKES"
    source_types: ["Patient"]
    target_types: ["Medication"]

  - name: "TREATS"
    source_types: ["Provider"]
    target_types: ["Patient"]

generic_to_domain_entity_map:
  person: "Patient"  # Default, can be overridden by context
  concept: "Diagnosis"
  event: "Encounter"
```

**Usage with Custom Transformer:**

```python
from akm.domain.transformer import BaseDomainTransformer
from akm.core.models import Entity

class HealthcareDomainTransformer(BaseDomainTransformer):
    """Custom transformer with healthcare-specific logic."""

    def map_entity(self, entity: Entity):
        # Custom logic: distinguish between Patient and Provider
        if entity.entity_type == "person":
            if "npi" in entity.properties or "specialty" in entity.properties:
                entity_dict = entity.model_dump()
                entity_dict["domain_type"] = "Provider"
                return Entity(**entity_dict)
            elif "patient_id" in entity.properties:
                entity_dict = entity.model_dump()
                entity_dict["domain_type"] = "Patient"
                return Entity(**entity_dict)

        return super().map_entity(entity)

    def get_extraction_prompt(self):
        return """Extract healthcare entities from clinical text:

        Entity types:
        - Patient: Names, patient IDs, demographics
        - Provider: Doctor names, NPI numbers, specialties
        - Diagnosis: ICD codes, condition names
        - Medication: Drug names, NDC codes, dosages
        - Encounter: Visit types, dates

        Ensure HIPAA compliance - do not extract SSN or financial info.
        """

# Usage
transformer = HealthcareDomainTransformer(schema_path="./schemas/healthcare.yaml")

doctor = Entity(
    name="Dr. Sarah Johnson",
    entity_type="person",
    properties={"npi": "1234567890", "specialty": "Cardiology"}
)

domain_doctor = transformer.map_entity(doctor)
print(domain_doctor.domain_type)  # "Provider"
```

## Integration with AKM

### Using with AKM Client

```python
from akm import AKM
from akm.domain.transformer import BaseDomainTransformer

# Create transformer
transformer = BaseDomainTransformer(schema_path="./schemas/software_engineering.yaml")

# Initialize AKM with transformer
akm = AKM.from_config("./akm.yaml")
akm.set_domain_transformer(transformer)
akm.connect()

# Entities are automatically mapped during ingestion
akm.ingest("./codebase/")

# Queries use domain context
result = akm.query("Which developers work on the auth component?")
```

### Using with Ingestion Pipeline

```python
from akm.ingestion.pipeline import IngestionPipeline
from akm.domain.transformer import BaseDomainTransformer

transformer = BaseDomainTransformer(schema_path="./schemas/research.yaml")

pipeline = IngestionPipeline(
    graph_backend=graph,
    vector_backend=vector,
    domain_transformer=transformer  # Entities mapped during ingestion
)

# Ingest research papers
stats = pipeline.ingest(paper_connector)
```

## Best Practices

1. **Start with Generic Types**: Use the built-in generic types (person, document, concept) and map to domain-specific types.

2. **Define Required Properties**: Mark essential properties as required for validation.

3. **Use Examples**: Provide examples in entity type definitions to improve type inference accuracy.

4. **Create Custom Transformers**: Extend `BaseDomainTransformer` for complex domain logic.

5. **Version Your Schemas**: Include version numbers to track schema evolution.

6. **Test Mappings**: Write unit tests for your mapping rules:

```python
def test_developer_mapping():
    transformer = BaseDomainTransformer(schema_path="./schemas/software_engineering.yaml")

    generic = Entity(name="John", entity_type="person")
    mapped = transformer.map_entity(generic)

    assert mapped.domain_type == "Developer"
```
