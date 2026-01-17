"""Base domain transformer implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from akm.core.interfaces import DomainTransformer, EmbeddingModel
from akm.core.models import (
    DomainSchema,
    Entity,
    EntityType,
    EntityTypeDefinition,
    Relationship,
    RelationshipType,
    RelationshipTypeDefinition,
)

logger = logging.getLogger(__name__)


class BaseDomainTransformer(DomainTransformer):
    """
    Base implementation of domain transformer.

    Domain transformers map generic entities and relationships to
    domain-specific types and provide domain context for queries.
    """

    def __init__(
        self,
        schema: Optional[DomainSchema] = None,
        schema_path: Optional[str] = None,
    ) -> None:
        """
        Initialize domain transformer.

        Args:
            schema: Domain schema object
            schema_path: Path to YAML schema file
        """
        self._schema: Optional[DomainSchema] = schema
        self._schema_path = schema_path
        self._embedding_model: Optional[EmbeddingModel] = None

        if schema_path and not schema:
            self._load_schema(schema_path)

    def _load_schema(self, path: str) -> None:
        """Load schema from YAML file."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            # Convert entity type definitions
            entity_types = []
            for et in data.get("entity_types", []):
                entity_types.append(EntityTypeDefinition(**et))

            # Convert relationship type definitions
            relationship_types = []
            for rt in data.get("relationship_types", []):
                relationship_types.append(RelationshipTypeDefinition(**rt))

            self._schema = DomainSchema(
                domain_name=data.get("domain_name", "generic"),
                version=data.get("version", "1.0.0"),
                description=data.get("description"),
                entity_types=entity_types,
                relationship_types=relationship_types,
                generic_to_domain_entity_map=data.get(
                    "generic_to_domain_entity_map", {}
                ),
                generic_to_domain_relationship_map=data.get(
                    "generic_to_domain_relationship_map", {}
                ),
                extraction_prompt_template=data.get("extraction_prompt_template"),
                query_context_template=data.get("query_context_template"),
            )
            logger.info(f"Loaded domain schema: {self._schema.domain_name}")
        except Exception as e:
            logger.error(f"Failed to load schema from {path}: {e}")
            raise

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        if self._schema:
            return self._schema.domain_name
        return "generic"

    @property
    def schema(self) -> DomainSchema:
        """Return the domain schema."""
        if not self._schema:
            self._schema = DomainSchema(domain_name="generic")
        return self._schema

    def map_entity(self, generic_entity: Entity) -> Optional[Entity]:
        """
        Map a generic entity to a domain-specific entity.

        Args:
            generic_entity: Entity with generic type

        Returns:
            Entity with domain-specific type, or original if no mapping
        """
        if not self._schema:
            return generic_entity

        # Get generic type string
        entity_type = generic_entity.entity_type
        if isinstance(entity_type, EntityType):
            entity_type = entity_type.value

        # Look up domain mapping
        domain_type = self._schema.generic_to_domain_entity_map.get(entity_type)

        if domain_type:
            # Create new entity with domain type
            entity_dict = generic_entity.model_dump()
            entity_dict["domain_type"] = domain_type

            # Apply domain-specific property transformations
            entity_dict = self._transform_entity_properties(entity_dict, domain_type)

            return Entity(**entity_dict)

        return generic_entity

    def _transform_entity_properties(
        self,
        entity_dict: Dict[str, Any],
        domain_type: str,
    ) -> Dict[str, Any]:
        """Apply domain-specific property transformations."""
        # Find the entity type definition
        type_def = self._get_entity_type_definition(domain_type)

        if type_def:
            properties = entity_dict.get("properties", {})

            # Validate required properties
            for req_prop in type_def.required_properties:
                if req_prop not in properties:
                    logger.warning(
                        f"Entity missing required property '{req_prop}' "
                        f"for domain type '{domain_type}'"
                    )

            entity_dict["properties"] = properties

        return entity_dict

    def _get_entity_type_definition(
        self, type_name: str
    ) -> Optional[EntityTypeDefinition]:
        """Get entity type definition by name."""
        if not self._schema:
            return None
        return next(
            (t for t in self._schema.entity_types if t.name == type_name),
            None,
        )

    def map_relationship(
        self,
        generic_relationship: Relationship,
    ) -> Optional[Relationship]:
        """
        Map a generic relationship to a domain-specific relationship.

        Args:
            generic_relationship: Relationship with generic type

        Returns:
            Relationship with domain-specific type, or original if no mapping
        """
        if not self._schema:
            return generic_relationship

        # Get generic type string
        rel_type = generic_relationship.relationship_type
        if isinstance(rel_type, RelationshipType):
            rel_type = rel_type.value

        # Look up domain mapping
        domain_type = self._schema.generic_to_domain_relationship_map.get(rel_type)

        if domain_type:
            rel_dict = generic_relationship.model_dump()
            rel_dict["domain_type"] = domain_type
            return Relationship(**rel_dict)

        return generic_relationship

    def get_prompt_context(self, query: str) -> str:
        """
        Get domain-specific context for LLM prompts.

        Args:
            query: The user's query

        Returns:
            Domain context string to inject into prompts
        """
        if not self._schema:
            return ""

        context_parts = []

        # Add domain description
        if self._schema.description:
            context_parts.append(f"Domain: {self._schema.domain_name}")
            context_parts.append(self._schema.description)

        # Add query context template if available
        if self._schema.query_context_template:
            context_parts.append(self._schema.query_context_template)

        # Add entity types context
        if self._schema.entity_types:
            entity_info = "Available entity types: " + ", ".join(
                t.name for t in self._schema.entity_types
            )
            context_parts.append(entity_info)

        # Add relationship types context
        if self._schema.relationship_types:
            rel_info = "Available relationship types: " + ", ".join(
                t.name for t in self._schema.relationship_types
            )
            context_parts.append(rel_info)

        return "\n".join(context_parts)

    def validate_entity(self, entity: Entity) -> bool:
        """
        Validate an entity against the domain schema.

        Args:
            entity: Entity to validate

        Returns:
            True if valid, False otherwise
        """
        if not self._schema:
            return True

        domain_type = entity.domain_type or entity.entity_type
        if isinstance(domain_type, EntityType):
            domain_type = domain_type.value

        # Find the type definition
        type_def = self._get_entity_type_definition(domain_type)

        if not type_def:
            return True  # Unknown types pass validation

        # Check required properties
        properties = entity.properties
        for req_prop in type_def.required_properties:
            if req_prop not in properties:
                logger.warning(
                    f"Entity '{entity.name}' missing required property: {req_prop}"
                )
                return False

        return True

    def get_embedding_model(self) -> Optional[EmbeddingModel]:
        """Get domain-specific embedding model if available."""
        return self._embedding_model

    def set_embedding_model(self, model: EmbeddingModel) -> None:
        """Set a domain-specific embedding model."""
        self._embedding_model = model

    def get_extraction_prompt(self) -> str:
        """Get the extraction prompt template for this domain."""
        if self._schema and self._schema.extraction_prompt_template:
            return self._schema.extraction_prompt_template

        # Default extraction prompt
        entity_types = (
            ", ".join(t.name for t in self._schema.entity_types)
            if self._schema
            else "person, organization, concept, document"
        )

        return f"""Extract entities and relationships from the text.

Entity types to extract: {entity_types}

For each entity found, provide:
1. The entity name/text
2. The entity type
3. A brief description
4. Confidence score (0-1)

For each relationship found, provide:
1. Source entity
2. Relationship type
3. Target entity
4. Confidence score (0-1)
"""

    def get_entity_types(self) -> List[str]:
        """Get list of domain entity type names."""
        if not self._schema:
            return []
        return [t.name for t in self._schema.entity_types]

    def get_relationship_types(self) -> List[str]:
        """Get list of domain relationship type names."""
        if not self._schema:
            return []
        return [t.name for t in self._schema.relationship_types]
