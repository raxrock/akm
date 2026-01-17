"""Core data models for the AKM framework."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Base entity types."""

    GENERIC = "generic"
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    DOCUMENT = "document"
    EVENT = "event"
    LOCATION = "location"
    CUSTOM = "custom"


class RelationshipType(str, Enum):
    """Base relationship types."""

    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    CREATED_BY = "CREATED_BY"
    REFERENCES = "REFERENCES"
    DEPENDS_ON = "DEPENDS_ON"
    SIMILAR_TO = "SIMILAR_TO"
    DERIVED_FROM = "DERIVED_FROM"
    CUSTOM = "CUSTOM"


class LinkStatus(str, Enum):
    """Link lifecycle status."""

    SOFT = "soft"  # Newly detected, unvalidated
    VALIDATING = "validating"  # Under validation
    VALIDATED = "validated"  # User-validated
    DECAYING = "decaying"  # Weight is decreasing
    ARCHIVED = "archived"  # Below threshold, not active


class AKMBaseModel(BaseModel):
    """Base model with common fields."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    }


class Entity(AKMBaseModel):
    """Represents a node in the knowledge graph."""

    name: str
    entity_type: Union[EntityType, str] = EntityType.GENERIC
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source_document_id: Optional[UUID] = None
    confidence: float = Field(default=1.0)
    domain_type: Optional[str] = None  # Domain-specific type override

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ExtractedEntity(BaseModel):
    """Entity extracted from text, before normalization."""

    text: str
    entity_type: Union[EntityType, str]
    start_char: int
    end_char: int
    confidence: float = 1.0
    source_text: Optional[str] = None


class Relationship(AKMBaseModel):
    """Represents an edge in the knowledge graph."""

    source_id: UUID
    target_id: UUID
    relationship_type: Union[RelationshipType, str] = RelationshipType.RELATED_TO
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bidirectional: bool = False
    domain_type: Optional[str] = None


class ExtractedRelationship(BaseModel):
    """Relationship extracted from text, before normalization."""

    source_entity: ExtractedEntity
    target_entity: ExtractedEntity
    relationship_type: Union[RelationshipType, str]
    confidence: float = 1.0
    source_text: Optional[str] = None


class LinkWeight(BaseModel):
    """Weight information for adaptive links."""

    value: float = Field(default=0.5, ge=0.0, le=1.0)
    initial_value: float = Field(default=0.5, ge=0.0, le=1.0)

    # Decay parameters
    decay_rate: float = Field(default=0.01, ge=0.0)  # Exponential decay rate per hour
    last_decay_at: datetime = Field(default_factory=datetime.utcnow)

    # Validation tracking
    validation_count: int = Field(default=0, ge=0)
    positive_validations: int = Field(default=0, ge=0)
    negative_validations: int = Field(default=0, ge=0)
    last_validated_at: Optional[datetime] = None

    # Thresholds
    promotion_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    demotion_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

    def apply_decay(self, time_delta_hours: float) -> float:
        """
        Apply exponential decay to the weight.

        Uses formula: weight * exp(-decay_rate * hours)

        Args:
            time_delta_hours: Hours elapsed since last decay

        Returns:
            New weight value after decay
        """
        decay_factor = math.exp(-self.decay_rate * time_delta_hours)
        self.value = max(0.0, self.value * decay_factor)
        self.last_decay_at = datetime.utcnow()
        return self.value

    def apply_validation(self, is_positive: bool, strength: float = 0.1) -> float:
        """
        Apply user validation to the weight.

        Args:
            is_positive: Whether the validation is positive (helpful) or negative
            strength: How much to adjust the weight (0-1)

        Returns:
            New weight value after validation
        """
        self.validation_count += 1
        if is_positive:
            self.positive_validations += 1
            self.value = min(1.0, self.value + strength)
        else:
            self.negative_validations += 1
            self.value = max(0.0, self.value - strength)
        self.last_validated_at = datetime.utcnow()
        return self.value

    def should_promote(self) -> bool:
        """Check if weight exceeds promotion threshold."""
        return self.value >= self.promotion_threshold

    def should_demote(self) -> bool:
        """Check if weight falls below demotion threshold."""
        return self.value <= self.demotion_threshold


class Link(AKMBaseModel):
    """Adaptive link connecting entities with weighted, evolving relationships."""

    source_id: UUID
    target_id: UUID
    link_type: str = "inferred"
    status: LinkStatus = LinkStatus.SOFT
    weight: LinkWeight = Field(default_factory=LinkWeight)

    # Pattern detection metadata
    pattern_source: Optional[str] = None  # What detected this pattern
    pattern_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Semantic context
    semantic_similarity: Optional[float] = None
    co_occurrence_count: int = Field(default=0, ge=0)

    # Graph context
    path_distance: Optional[int] = None  # Shortest path in graph
    common_neighbors: int = Field(default=0, ge=0)

    def should_promote(self) -> bool:
        """Check if link should be promoted to validated status."""
        return self.weight.should_promote()

    def should_demote(self) -> bool:
        """Check if link should be demoted/archived."""
        return self.weight.should_demote()

    def increment_co_occurrence(self) -> None:
        """Increment co-occurrence count and adjust weight."""
        self.co_occurrence_count += 1
        # Small weight boost for co-occurrence
        boost = min(0.05, 0.01 * self.co_occurrence_count)
        self.weight.value = min(1.0, self.weight.value + boost)


class Document(AKMBaseModel):
    """Represents a source document."""

    content: str
    title: Optional[str] = None
    source: str  # File path, URL, etc.
    source_type: str  # "file", "slack", "email", etc.
    mime_type: Optional[str] = None
    encoding: str = "utf-8"
    content_hash: Optional[str] = None  # Content hash for deduplication


class Chunk(AKMBaseModel):
    """A chunk of a document for embedding."""

    content: str
    document_id: UUID
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    token_count: Optional[int] = None


class RawDocument(BaseModel):
    """Raw document from a connector before processing."""

    content: Union[str, bytes]
    source: str
    source_type: str
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result from a search operation."""

    document_id: str
    chunk_id: Optional[str] = None
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entity_matches: List[Entity] = Field(default_factory=list)


class ContextItem(BaseModel):
    """Context item for LLM queries."""

    content: str
    source: str
    relevance_score: float = 1.0
    entity_context: Optional[List[Entity]] = None


class QueryResult(BaseModel):
    """Result from a query engine operation."""

    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    entities_involved: List[Entity] = Field(default_factory=list)
    relationships_involved: List[Relationship] = Field(default_factory=list)
    links_involved: List[Link] = Field(default_factory=list)
    reasoning_path: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    execution_time_ms: float = 0.0


class TraversalResult(BaseModel):
    """Result from a graph traversal operation."""

    start_entity: Entity
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    links: List[Link] = Field(default_factory=list)
    paths: List[List[UUID]] = Field(default_factory=list)
    depth_reached: int = 0


class GraphData(BaseModel):
    """Data structure for GNN operations."""

    node_features: List[List[float]]
    edge_index: List[List[int]]  # Shape: [2, num_edges]
    edge_features: Optional[List[List[float]]] = None
    node_labels: Optional[List[int]] = None
    edge_labels: Optional[List[int]] = None
    node_ids: List[UUID] = Field(default_factory=list)


class NodeEmbeddings(BaseModel):
    """Node embeddings from GNN."""

    embeddings: List[List[float]]
    node_ids: List[UUID]
    dimension: int


class TrainingResult(BaseModel):
    """Result from GNN training."""

    epochs_completed: int
    final_loss: float
    best_loss: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    training_time_seconds: float = 0.0


class EntityTypeDefinition(BaseModel):
    """Definition of an entity type in a domain schema."""

    name: str
    base_type: EntityType = EntityType.CUSTOM
    description: Optional[str] = None
    properties: Dict[str, str] = Field(default_factory=dict)  # name -> type
    required_properties: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class RelationshipTypeDefinition(BaseModel):
    """Definition of a relationship type in a domain schema."""

    name: str
    base_type: RelationshipType = RelationshipType.CUSTOM
    description: Optional[str] = None
    source_types: List[str] = Field(default_factory=list)
    target_types: List[str] = Field(default_factory=list)
    properties: Dict[str, str] = Field(default_factory=dict)
    bidirectional: bool = False


class DomainSchema(BaseModel):
    """Complete domain schema definition."""

    domain_name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    entity_types: List[EntityTypeDefinition] = Field(default_factory=list)
    relationship_types: List[RelationshipTypeDefinition] = Field(default_factory=list)

    # Mapping rules
    generic_to_domain_entity_map: Dict[str, str] = Field(default_factory=dict)
    generic_to_domain_relationship_map: Dict[str, str] = Field(default_factory=dict)

    # Prompt templates
    extraction_prompt_template: Optional[str] = None
    query_context_template: Optional[str] = None
