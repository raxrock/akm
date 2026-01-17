"""Neo4j graph backend module."""

from akm.graph.neo4j.client import Neo4jBackend
from akm.graph.neo4j.query_builder import (
    ComparisonOperator,
    Condition,
    ConditionGroup,
    CypherQueryBuilder,
    LogicalOperator,
    SortCriteria,
    SortDirection,
    match_entities_by_type,
    match_entity_by_id,
    match_neighbors,
    search_entities_by_text,
)
from akm.graph.neo4j.repository import Neo4jRepository

__all__ = [
    # Client
    "Neo4jBackend",
    # Repository
    "Neo4jRepository",
    # Query Builder
    "CypherQueryBuilder",
    "Condition",
    "ConditionGroup",
    "ComparisonOperator",
    "LogicalOperator",
    "SortDirection",
    "SortCriteria",
    # Convenience functions
    "match_entity_by_id",
    "match_entities_by_type",
    "match_neighbors",
    "search_entities_by_text",
]
