"""Domain schema utilities."""

from akm.domain.schema.loader import (
    load_schema_from_yaml,
    merge_schemas,
    parse_schema_dict,
    save_schema_to_yaml,
    validate_schema,
)
from akm.domain.schema.mapper import (
    MappingResult,
    MappingRule,
    SchemaMapper,
    TypeInferencer,
)

__all__ = [
    # Schema loading
    "load_schema_from_yaml",
    "save_schema_to_yaml",
    "parse_schema_dict",
    "validate_schema",
    "merge_schemas",
    # Schema mapping
    "SchemaMapper",
    "MappingRule",
    "MappingResult",
    "TypeInferencer",
]
