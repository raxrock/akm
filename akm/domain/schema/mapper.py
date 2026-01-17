"""Generic-to-domain entity and relationship mapping."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


@dataclass
class MappingRule:
    """A rule for mapping generic types to domain types."""

    source_type: str
    target_type: str
    conditions: List[Callable[[Union[Entity, Relationship]], bool]] = field(default_factory=list)
    property_transforms: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    priority: int = 0  # Higher priority rules are checked first

    def matches(self, obj: Union[Entity, Relationship]) -> bool:
        """Check if this rule matches the object."""
        # Check type match
        obj_type = self._get_type_string(obj)
        if obj_type != self.source_type:
            return False

        # Check all conditions
        return all(cond(obj) for cond in self.conditions)

    def _get_type_string(self, obj: Union[Entity, Relationship]) -> str:
        """Get type as string."""
        if isinstance(obj, Entity):
            type_val = obj.entity_type
        else:
            type_val = obj.relationship_type

        if hasattr(type_val, "value"):
            return type_val.value
        return str(type_val)


@dataclass
class MappingResult:
    """Result of a mapping operation."""

    original: Union[Entity, Relationship]
    mapped: Union[Entity, Relationship]
    rule_applied: Optional[MappingRule] = None
    transformations: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class SchemaMapper:
    """
    Maps generic entities and relationships to domain-specific types.

    The mapper uses rules to:
    1. Convert generic types to domain types
    2. Transform properties according to schema
    3. Validate against domain constraints
    4. Apply domain-specific enrichments
    """

    def __init__(
        self,
        schema: Optional[DomainSchema] = None,
    ) -> None:
        """
        Initialize the schema mapper.

        Args:
            schema: Domain schema for mapping
        """
        self._schema = schema
        self._entity_rules: List[MappingRule] = []
        self._relationship_rules: List[MappingRule] = []
        self._type_definitions: Dict[str, EntityTypeDefinition] = {}
        self._rel_type_definitions: Dict[str, RelationshipTypeDefinition] = {}

        if schema:
            self._load_schema(schema)

    def _load_schema(self, schema: DomainSchema) -> None:
        """Load mapping rules from schema."""
        # Build type definition lookup
        for type_def in schema.entity_types:
            self._type_definitions[type_def.name] = type_def

        for rel_def in schema.relationship_types:
            self._rel_type_definitions[rel_def.name] = rel_def

        # Create rules from generic-to-domain maps
        for generic_type, domain_type in schema.generic_to_domain_entity_map.items():
            self._entity_rules.append(
                MappingRule(
                    source_type=generic_type,
                    target_type=domain_type,
                    priority=1,
                )
            )

        for generic_type, domain_type in schema.generic_to_domain_relationship_map.items():
            self._relationship_rules.append(
                MappingRule(
                    source_type=generic_type,
                    target_type=domain_type,
                    priority=1,
                )
            )

        # Sort rules by priority (descending)
        self._entity_rules.sort(key=lambda r: r.priority, reverse=True)
        self._relationship_rules.sort(key=lambda r: r.priority, reverse=True)

    def add_entity_rule(self, rule: MappingRule) -> None:
        """Add an entity mapping rule."""
        self._entity_rules.append(rule)
        self._entity_rules.sort(key=lambda r: r.priority, reverse=True)

    def add_relationship_rule(self, rule: MappingRule) -> None:
        """Add a relationship mapping rule."""
        self._relationship_rules.append(rule)
        self._relationship_rules.sort(key=lambda r: r.priority, reverse=True)

    def map_entity(self, entity: Entity) -> MappingResult:
        """
        Map a generic entity to a domain-specific entity.

        Args:
            entity: Entity to map

        Returns:
            Mapping result with mapped entity
        """
        # Find matching rule
        matching_rule = None
        for rule in self._entity_rules:
            if rule.matches(entity):
                matching_rule = rule
                break

        if not matching_rule:
            # No mapping needed, return original
            return MappingResult(
                original=entity,
                mapped=entity,
                success=True,
            )

        try:
            # Create mapped entity
            entity_dict = entity.model_dump()
            transformations = []

            # Apply domain type
            entity_dict["domain_type"] = matching_rule.target_type
            transformations.append(f"Set domain_type to '{matching_rule.target_type}'")

            # Apply property transforms
            for prop_name, transform in matching_rule.property_transforms.items():
                if prop_name in entity_dict.get("properties", {}):
                    old_value = entity_dict["properties"][prop_name]
                    entity_dict["properties"][prop_name] = transform(old_value)
                    transformations.append(f"Transformed property '{prop_name}'")

            # Validate against type definition
            type_def = self._type_definitions.get(matching_rule.target_type)
            if type_def:
                validation_result = self._validate_entity_properties(
                    entity_dict, type_def
                )
                if not validation_result[0]:
                    return MappingResult(
                        original=entity,
                        mapped=entity,
                        rule_applied=matching_rule,
                        success=False,
                        error=validation_result[1],
                    )

            mapped_entity = Entity(**entity_dict)

            return MappingResult(
                original=entity,
                mapped=mapped_entity,
                rule_applied=matching_rule,
                transformations=transformations,
                success=True,
            )

        except Exception as e:
            logger.error(f"Entity mapping failed: {e}")
            return MappingResult(
                original=entity,
                mapped=entity,
                rule_applied=matching_rule,
                success=False,
                error=str(e),
            )

    def map_relationship(self, relationship: Relationship) -> MappingResult:
        """
        Map a generic relationship to a domain-specific relationship.

        Args:
            relationship: Relationship to map

        Returns:
            Mapping result with mapped relationship
        """
        # Find matching rule
        matching_rule = None
        for rule in self._relationship_rules:
            if rule.matches(relationship):
                matching_rule = rule
                break

        if not matching_rule:
            return MappingResult(
                original=relationship,
                mapped=relationship,
                success=True,
            )

        try:
            rel_dict = relationship.model_dump()
            transformations = []

            # Apply domain type
            rel_dict["domain_type"] = matching_rule.target_type
            transformations.append(f"Set domain_type to '{matching_rule.target_type}'")

            # Apply property transforms
            for prop_name, transform in matching_rule.property_transforms.items():
                if prop_name in rel_dict.get("properties", {}):
                    rel_dict["properties"][prop_name] = transform(rel_dict["properties"][prop_name])
                    transformations.append(f"Transformed property '{prop_name}'")

            mapped_relationship = Relationship(**rel_dict)

            return MappingResult(
                original=relationship,
                mapped=mapped_relationship,
                rule_applied=matching_rule,
                transformations=transformations,
                success=True,
            )

        except Exception as e:
            logger.error(f"Relationship mapping failed: {e}")
            return MappingResult(
                original=relationship,
                mapped=relationship,
                rule_applied=matching_rule,
                success=False,
                error=str(e),
            )

    def map_entities_batch(self, entities: List[Entity]) -> List[MappingResult]:
        """Map multiple entities."""
        return [self.map_entity(e) for e in entities]

    def map_relationships_batch(
        self,
        relationships: List[Relationship],
    ) -> List[MappingResult]:
        """Map multiple relationships."""
        return [self.map_relationship(r) for r in relationships]

    def infer_entity_type(
        self,
        name: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Infer the domain entity type from name and context.

        Args:
            name: Entity name
            description: Optional description
            properties: Optional properties

        Returns:
            Inferred domain type
        """
        if not self._schema:
            return "generic"

        name_lower = name.lower()
        desc_lower = (description or "").lower()

        # Check against type definitions
        for type_def in self._schema.entity_types:
            # Check examples
            for example in type_def.examples:
                if example.lower() in name_lower or name_lower in example.lower():
                    return type_def.name

            # Check description keywords
            if type_def.description:
                keywords = type_def.description.lower().split()
                if any(kw in name_lower or kw in desc_lower for kw in keywords if len(kw) > 3):
                    return type_def.name

        # Check generic-to-domain map
        for generic_type, domain_type in self._schema.generic_to_domain_entity_map.items():
            if generic_type.lower() in name_lower or generic_type.lower() in desc_lower:
                return domain_type

        return "generic"

    def get_required_properties(self, domain_type: str) -> List[str]:
        """Get required properties for a domain type."""
        type_def = self._type_definitions.get(domain_type)
        if type_def:
            return type_def.required_properties
        return []

    def get_property_types(self, domain_type: str) -> Dict[str, str]:
        """Get property type definitions for a domain type."""
        type_def = self._type_definitions.get(domain_type)
        if type_def:
            return type_def.properties
        return {}

    def validate_entity(self, entity: Entity) -> Tuple[bool, Optional[str]]:
        """
        Validate an entity against domain schema.

        Args:
            entity: Entity to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        domain_type = entity.domain_type or entity.entity_type
        if hasattr(domain_type, "value"):
            domain_type = domain_type.value

        type_def = self._type_definitions.get(domain_type)
        if not type_def:
            return True, None  # Unknown types pass validation

        return self._validate_entity_properties(entity.model_dump(), type_def)

    def _validate_entity_properties(
        self,
        entity_dict: Dict[str, Any],
        type_def: EntityTypeDefinition,
    ) -> Tuple[bool, Optional[str]]:
        """Validate entity properties against type definition."""
        properties = entity_dict.get("properties", {})

        # Check required properties
        for req_prop in type_def.required_properties:
            if req_prop not in properties:
                return False, f"Missing required property: {req_prop}"

        # Check property types (basic validation)
        for prop_name, expected_type in type_def.properties.items():
            if prop_name in properties:
                value = properties[prop_name]
                if not self._check_property_type(value, expected_type):
                    return False, f"Property '{prop_name}' has wrong type (expected {expected_type})"

        return True, None

    def _check_property_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type."""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "int": lambda v: isinstance(v, int),
            "integer": lambda v: isinstance(v, int),
            "float": lambda v: isinstance(v, (int, float)),
            "number": lambda v: isinstance(v, (int, float)),
            "bool": lambda v: isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "list": lambda v: isinstance(v, list),
            "array": lambda v: isinstance(v, list),
            "dict": lambda v: isinstance(v, dict),
            "object": lambda v: isinstance(v, dict),
        }

        checker = type_checks.get(expected_type.lower())
        if checker:
            return checker(value)
        return True  # Unknown types pass

    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about mapping rules."""
        return {
            "entity_rules": len(self._entity_rules),
            "relationship_rules": len(self._relationship_rules),
            "entity_type_definitions": len(self._type_definitions),
            "relationship_type_definitions": len(self._rel_type_definitions),
            "domain_name": self._schema.domain_name if self._schema else "none",
        }


class TypeInferencer:
    """
    Infers entity and relationship types from text and context.

    Uses patterns and heuristics to determine types
    when not explicitly provided.
    """

    def __init__(
        self,
        schema: Optional[DomainSchema] = None,
    ) -> None:
        """
        Initialize the type inferencer.

        Args:
            schema: Optional domain schema for type hints
        """
        self._schema = schema

        # Default patterns for common entity types
        self._entity_patterns = {
            "person": [
                r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Jr\.|Sr\.|III|IV)\b",
            ],
            "organization": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|LLC|Corp\.|Company|Ltd\.)\b",
                r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b",  # Acronyms
            ],
            "location": [
                r"\b(?:City of|Town of|State of)\s+[A-Z][a-z]+\b",
                r"\b[A-Z][a-z]+,\s+[A-Z]{2}\b",  # City, ST format
            ],
            "document": [
                r"\b[a-z_]+\.(?:md|txt|pdf|doc|docx)\b",
                r"\bREADME\b",
            ],
            "concept": [
                r"\b(?:concept of|theory of|principle of)\s+[a-z]+\b",
            ],
        }

        # Load patterns from schema if available
        if schema:
            self._load_schema_patterns(schema)

    def _load_schema_patterns(self, schema: DomainSchema) -> None:
        """Load type patterns from schema examples."""
        for type_def in schema.entity_types:
            if type_def.examples:
                # Create patterns from examples
                patterns = [re.escape(ex) for ex in type_def.examples]
                self._entity_patterns[type_def.name] = patterns

    def infer_entity_type(
        self,
        name: str,
        context: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Infer entity type from name and context.

        Args:
            name: Entity name
            context: Optional surrounding text context

        Returns:
            Tuple of (inferred_type, confidence)
        """
        text_to_check = f"{name} {context or ''}"

        for entity_type, patterns in self._entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_to_check, re.IGNORECASE):
                    return entity_type, 0.8

        # Check for common indicators
        name_lower = name.lower()

        if any(indicator in name_lower for indicator in ["service", "api", "module", "component"]):
            return "component", 0.7

        if any(indicator in name_lower for indicator in ["database", "db", "postgresql", "mongodb"]):
            return "technology", 0.7

        if any(indicator in name_lower for indicator in ["team", "department", "group"]):
            return "organization", 0.6

        return "generic", 0.5

    def infer_relationship_type(
        self,
        source_type: str,
        target_type: str,
        context: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Infer relationship type from entity types and context.

        Args:
            source_type: Source entity type
            target_type: Target entity type
            context: Optional text context

        Returns:
            Tuple of (inferred_type, confidence)
        """
        # Common relationship patterns
        patterns = {
            ("person", "organization"): ("WORKS_AT", 0.7),
            ("person", "document"): ("AUTHORED", 0.6),
            ("component", "component"): ("DEPENDS_ON", 0.6),
            ("component", "technology"): ("USES", 0.7),
            ("document", "concept"): ("DESCRIBES", 0.6),
        }

        key = (source_type, target_type)
        if key in patterns:
            return patterns[key]

        # Check context for relationship indicators
        if context:
            context_lower = context.lower()
            if "created by" in context_lower or "authored by" in context_lower:
                return "CREATED_BY", 0.8
            if "depends on" in context_lower or "requires" in context_lower:
                return "DEPENDS_ON", 0.8
            if "uses" in context_lower or "utilizes" in context_lower:
                return "USES", 0.8
            if "part of" in context_lower or "belongs to" in context_lower:
                return "PART_OF", 0.8

        return "RELATED_TO", 0.5
