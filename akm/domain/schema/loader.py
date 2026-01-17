"""Domain schema loader utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from akm.core.exceptions import SchemaError
from akm.core.models import (
    DomainSchema,
    EntityTypeDefinition,
    RelationshipTypeDefinition,
)

logger = logging.getLogger(__name__)


def load_schema_from_yaml(path: Union[str, Path]) -> DomainSchema:
    """
    Load a domain schema from a YAML file.

    Args:
        path: Path to YAML schema file

    Returns:
        DomainSchema object

    Raises:
        SchemaError: If schema loading fails
    """
    path = Path(path)

    if not path.exists():
        raise SchemaError(f"Schema file not found: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return parse_schema_dict(data)

    except yaml.YAMLError as e:
        raise SchemaError(f"Invalid YAML in schema file: {e}")
    except Exception as e:
        raise SchemaError(f"Failed to load schema: {e}")


def parse_schema_dict(data: Dict) -> DomainSchema:
    """
    Parse a schema dictionary into a DomainSchema object.

    Args:
        data: Dictionary containing schema data

    Returns:
        DomainSchema object
    """
    if not data:
        raise SchemaError("Empty schema data")

    # Parse entity types
    entity_types = []
    for et_data in data.get("entity_types", []):
        try:
            entity_types.append(EntityTypeDefinition(**et_data))
        except Exception as e:
            logger.warning(f"Failed to parse entity type: {e}")

    # Parse relationship types
    relationship_types = []
    for rt_data in data.get("relationship_types", []):
        try:
            relationship_types.append(RelationshipTypeDefinition(**rt_data))
        except Exception as e:
            logger.warning(f"Failed to parse relationship type: {e}")

    return DomainSchema(
        domain_name=data.get("domain_name", "generic"),
        version=data.get("version", "1.0.0"),
        description=data.get("description"),
        entity_types=entity_types,
        relationship_types=relationship_types,
        generic_to_domain_entity_map=data.get("generic_to_domain_entity_map", {}),
        generic_to_domain_relationship_map=data.get(
            "generic_to_domain_relationship_map", {}
        ),
        extraction_prompt_template=data.get("extraction_prompt_template"),
        query_context_template=data.get("query_context_template"),
    )


def save_schema_to_yaml(schema: DomainSchema, path: Union[str, Path]) -> None:
    """
    Save a domain schema to a YAML file.

    Args:
        schema: DomainSchema object
        path: Path to save YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "domain_name": schema.domain_name,
        "version": schema.version,
        "description": schema.description,
        "entity_types": [et.model_dump() for et in schema.entity_types],
        "relationship_types": [rt.model_dump() for rt in schema.relationship_types],
        "generic_to_domain_entity_map": schema.generic_to_domain_entity_map,
        "generic_to_domain_relationship_map": schema.generic_to_domain_relationship_map,
        "extraction_prompt_template": schema.extraction_prompt_template,
        "query_context_template": schema.query_context_template,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved schema to {path}")


def validate_schema(schema: DomainSchema) -> list[str]:
    """
    Validate a domain schema.

    Args:
        schema: DomainSchema to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not schema.domain_name:
        errors.append("Schema must have a domain_name")

    # Check entity type definitions
    entity_names = set()
    for et in schema.entity_types:
        if not et.name:
            errors.append("Entity type must have a name")
        elif et.name in entity_names:
            errors.append(f"Duplicate entity type name: {et.name}")
        else:
            entity_names.add(et.name)

    # Check relationship type definitions
    rel_names = set()
    for rt in schema.relationship_types:
        if not rt.name:
            errors.append("Relationship type must have a name")
        elif rt.name in rel_names:
            errors.append(f"Duplicate relationship type name: {rt.name}")
        else:
            rel_names.add(rt.name)

        # Check source/target types reference valid entity types
        for source_type in rt.source_types:
            if source_type not in entity_names:
                errors.append(
                    f"Relationship '{rt.name}' references unknown source type: {source_type}"
                )
        for target_type in rt.target_types:
            if target_type not in entity_names:
                errors.append(
                    f"Relationship '{rt.name}' references unknown target type: {target_type}"
                )

    # Check mappings reference valid types
    for generic_type, domain_type in schema.generic_to_domain_entity_map.items():
        if domain_type not in entity_names:
            errors.append(
                f"Entity mapping '{generic_type}' -> '{domain_type}' references unknown type"
            )

    return errors


def merge_schemas(base: DomainSchema, extension: DomainSchema) -> DomainSchema:
    """
    Merge two schemas, with extension taking precedence.

    Args:
        base: Base schema
        extension: Extension schema to merge in

    Returns:
        Merged DomainSchema
    """
    # Merge entity types
    entity_types = {et.name: et for et in base.entity_types}
    for et in extension.entity_types:
        entity_types[et.name] = et

    # Merge relationship types
    relationship_types = {rt.name: rt for rt in base.relationship_types}
    for rt in extension.relationship_types:
        relationship_types[rt.name] = rt

    # Merge mappings
    entity_map = {**base.generic_to_domain_entity_map}
    entity_map.update(extension.generic_to_domain_entity_map)

    rel_map = {**base.generic_to_domain_relationship_map}
    rel_map.update(extension.generic_to_domain_relationship_map)

    return DomainSchema(
        domain_name=extension.domain_name or base.domain_name,
        version=extension.version or base.version,
        description=extension.description or base.description,
        entity_types=list(entity_types.values()),
        relationship_types=list(relationship_types.values()),
        generic_to_domain_entity_map=entity_map,
        generic_to_domain_relationship_map=rel_map,
        extraction_prompt_template=(
            extension.extraction_prompt_template or base.extraction_prompt_template
        ),
        query_context_template=(
            extension.query_context_template or base.query_context_template
        ),
    )
