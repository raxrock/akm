"""Cypher query builder for Neo4j operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ComparisonOperator(str, Enum):
    """Comparison operators for query conditions."""

    EQUALS = "="
    NOT_EQUALS = "<>"
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUALS = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUALS = "<="
    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS WITH"
    ENDS_WITH = "ENDS WITH"
    IN = "IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    REGEX = "=~"


class LogicalOperator(str, Enum):
    """Logical operators for combining conditions."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"


class SortDirection(str, Enum):
    """Sort direction for ORDER BY."""

    ASC = "ASC"
    DESC = "DESC"


@dataclass
class Condition:
    """Represents a query condition."""

    property_path: str
    operator: ComparisonOperator
    value: Any = None
    param_name: Optional[str] = None

    def to_cypher(self, alias: str = "n") -> tuple[str, Dict[str, Any]]:
        """
        Convert condition to Cypher clause.

        Args:
            alias: Node alias in the query

        Returns:
            Tuple of (cypher_string, parameters)
        """
        param_name = self.param_name or self.property_path.replace(".", "_")

        if self.operator in (ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL):
            return f"{alias}.{self.property_path} {self.operator.value}", {}

        if self.operator == ComparisonOperator.IN:
            return f"{alias}.{self.property_path} IN ${param_name}", {
                param_name: self.value
            }

        return f"{alias}.{self.property_path} {self.operator.value} ${param_name}", {
            param_name: self.value
        }


@dataclass
class ConditionGroup:
    """Represents a group of conditions combined with a logical operator."""

    conditions: List[Union[Condition, "ConditionGroup"]] = field(default_factory=list)
    operator: LogicalOperator = LogicalOperator.AND

    def to_cypher(self, alias: str = "n") -> tuple[str, Dict[str, Any]]:
        """Convert condition group to Cypher clause."""
        if not self.conditions:
            return "", {}

        parts = []
        params = {}

        for cond in self.conditions:
            cypher, cond_params = cond.to_cypher(alias)
            parts.append(cypher)
            params.update(cond_params)

        joined = f" {self.operator.value} ".join(f"({p})" for p in parts)
        return f"({joined})", params


@dataclass
class SortCriteria:
    """Represents a sort criterion."""

    property_path: str
    direction: SortDirection = SortDirection.ASC

    def to_cypher(self, alias: str = "n") -> str:
        """Convert to Cypher ORDER BY clause."""
        return f"{alias}.{self.property_path} {self.direction.value}"


class CypherQueryBuilder:
    """
    Fluent builder for constructing Cypher queries.

    Example usage:
        query = (
            CypherQueryBuilder()
            .match("Entity", alias="e")
            .where("entity_type", ComparisonOperator.EQUALS, "person")
            .where("confidence", ComparisonOperator.GREATER_THAN, 0.5)
            .order_by("name")
            .limit(10)
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize the query builder."""
        self._match_patterns: List[str] = []
        self._optional_match_patterns: List[str] = []
        self._where_conditions: List[Union[Condition, ConditionGroup]] = []
        self._return_items: List[str] = []
        self._order_by: List[SortCriteria] = []
        self._skip: Optional[int] = None
        self._limit: Optional[int] = None
        self._with_clauses: List[str] = []
        self._create_patterns: List[str] = []
        self._set_clauses: List[str] = []
        self._delete_items: List[str] = []
        self._detach_delete: bool = False
        self._parameters: Dict[str, Any] = {}
        self._current_alias: str = "n"

    def match(
        self,
        label: str,
        alias: str = "n",
        properties: Optional[Dict[str, Any]] = None,
    ) -> "CypherQueryBuilder":
        """
        Add a MATCH clause.

        Args:
            label: Node label
            alias: Node alias
            properties: Optional inline properties to match

        Returns:
            Self for chaining
        """
        self._current_alias = alias
        if properties:
            props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
            self._match_patterns.append(f"({alias}:{label} {{{props_str}}})")
            self._parameters.update(properties)
        else:
            self._match_patterns.append(f"({alias}:{label})")
        return self

    def match_relationship(
        self,
        source_alias: str,
        target_alias: str,
        rel_type: Optional[str] = None,
        rel_alias: str = "r",
        direction: str = "outgoing",
        min_hops: int = 1,
        max_hops: int = 1,
    ) -> "CypherQueryBuilder":
        """
        Add a relationship pattern to MATCH.

        Args:
            source_alias: Source node alias
            target_alias: Target node alias
            rel_type: Optional relationship type
            rel_alias: Relationship alias
            direction: "outgoing", "incoming", or "both"
            min_hops: Minimum hops for variable length paths
            max_hops: Maximum hops for variable length paths

        Returns:
            Self for chaining
        """
        type_str = f":{rel_type}" if rel_type else ""

        if min_hops == 1 and max_hops == 1:
            rel_pattern = f"[{rel_alias}{type_str}]"
        else:
            rel_pattern = f"[{rel_alias}{type_str}*{min_hops}..{max_hops}]"

        if direction == "outgoing":
            pattern = f"({source_alias})-{rel_pattern}->({target_alias})"
        elif direction == "incoming":
            pattern = f"({source_alias})<-{rel_pattern}-({target_alias})"
        else:
            pattern = f"({source_alias})-{rel_pattern}-({target_alias})"

        self._match_patterns.append(pattern)
        return self

    def optional_match(
        self,
        label: str,
        alias: str = "n",
    ) -> "CypherQueryBuilder":
        """
        Add an OPTIONAL MATCH clause.

        Args:
            label: Node label
            alias: Node alias

        Returns:
            Self for chaining
        """
        self._optional_match_patterns.append(f"({alias}:{label})")
        return self

    def where(
        self,
        property_path: str,
        operator: ComparisonOperator = ComparisonOperator.EQUALS,
        value: Any = None,
        param_name: Optional[str] = None,
    ) -> "CypherQueryBuilder":
        """
        Add a WHERE condition.

        Args:
            property_path: Property path (e.g., "name", "properties.key")
            operator: Comparison operator
            value: Value to compare against
            param_name: Optional parameter name override

        Returns:
            Self for chaining
        """
        condition = Condition(
            property_path=property_path,
            operator=operator,
            value=value,
            param_name=param_name,
        )
        self._where_conditions.append(condition)
        return self

    def where_group(
        self,
        conditions: List[tuple],
        operator: LogicalOperator = LogicalOperator.AND,
    ) -> "CypherQueryBuilder":
        """
        Add a group of conditions.

        Args:
            conditions: List of (property, operator, value) tuples
            operator: Logical operator to combine conditions

        Returns:
            Self for chaining
        """
        cond_objects = [
            Condition(property_path=c[0], operator=c[1], value=c[2] if len(c) > 2 else None)
            for c in conditions
        ]
        group = ConditionGroup(conditions=cond_objects, operator=operator)
        self._where_conditions.append(group)
        return self

    def where_raw(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> "CypherQueryBuilder":
        """
        Add a raw WHERE condition.

        Args:
            cypher: Raw Cypher condition string
            params: Parameters for the condition

        Returns:
            Self for chaining
        """
        # Store as a special condition that outputs raw cypher
        self._where_conditions.append(_RawCondition(cypher, params or {}))
        return self

    def return_items(self, *items: str) -> "CypherQueryBuilder":
        """
        Set RETURN items.

        Args:
            items: Items to return (e.g., "n", "n.name AS name")

        Returns:
            Self for chaining
        """
        self._return_items.extend(items)
        return self

    def return_node(self, alias: Optional[str] = None) -> "CypherQueryBuilder":
        """
        Return a node by alias.

        Args:
            alias: Node alias (defaults to current alias)

        Returns:
            Self for chaining
        """
        self._return_items.append(alias or self._current_alias)
        return self

    def return_count(self, alias: Optional[str] = None, as_name: str = "count") -> "CypherQueryBuilder":
        """
        Return count of nodes.

        Args:
            alias: Node alias (defaults to current alias)
            as_name: Name for the count result

        Returns:
            Self for chaining
        """
        self._return_items.append(f"count({alias or self._current_alias}) AS {as_name}")
        return self

    def return_distinct(self, alias: Optional[str] = None) -> "CypherQueryBuilder":
        """
        Return distinct nodes.

        Args:
            alias: Node alias (defaults to current alias)

        Returns:
            Self for chaining
        """
        self._return_items.append(f"DISTINCT {alias or self._current_alias}")
        return self

    def order_by(
        self,
        property_path: str,
        direction: SortDirection = SortDirection.ASC,
        alias: Optional[str] = None,
    ) -> "CypherQueryBuilder":
        """
        Add ORDER BY clause.

        Args:
            property_path: Property to sort by
            direction: Sort direction
            alias: Node alias (defaults to current alias)

        Returns:
            Self for chaining
        """
        criteria = SortCriteria(property_path=property_path, direction=direction)
        criteria._alias = alias or self._current_alias
        self._order_by.append(criteria)
        return self

    def skip(self, count: int) -> "CypherQueryBuilder":
        """
        Add SKIP clause.

        Args:
            count: Number of results to skip

        Returns:
            Self for chaining
        """
        self._skip = count
        return self

    def limit(self, count: int) -> "CypherQueryBuilder":
        """
        Add LIMIT clause.

        Args:
            count: Maximum number of results

        Returns:
            Self for chaining
        """
        self._limit = count
        return self

    def with_clause(self, *items: str) -> "CypherQueryBuilder":
        """
        Add WITH clause.

        Args:
            items: Items to pass through

        Returns:
            Self for chaining
        """
        self._with_clauses.append(", ".join(items))
        return self

    def create(
        self,
        label: str,
        alias: str = "n",
        properties: Optional[Dict[str, Any]] = None,
    ) -> "CypherQueryBuilder":
        """
        Add CREATE clause for a node.

        Args:
            label: Node label
            alias: Node alias
            properties: Node properties

        Returns:
            Self for chaining
        """
        if properties:
            props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
            self._create_patterns.append(f"({alias}:{label} {{{props_str}}})")
            self._parameters.update(properties)
        else:
            self._create_patterns.append(f"({alias}:{label})")
        return self

    def set(self, property_path: str, value: Any, alias: Optional[str] = None) -> "CypherQueryBuilder":
        """
        Add SET clause.

        Args:
            property_path: Property to set
            value: Value to set
            alias: Node alias (defaults to current alias)

        Returns:
            Self for chaining
        """
        param_name = property_path.replace(".", "_")
        node_alias = alias or self._current_alias
        self._set_clauses.append(f"{node_alias}.{property_path} = ${param_name}")
        self._parameters[param_name] = value
        return self

    def delete(self, *aliases: str, detach: bool = False) -> "CypherQueryBuilder":
        """
        Add DELETE clause.

        Args:
            aliases: Aliases to delete
            detach: Whether to use DETACH DELETE

        Returns:
            Self for chaining
        """
        self._delete_items.extend(aliases)
        self._detach_delete = detach
        return self

    def set_parameter(self, name: str, value: Any) -> "CypherQueryBuilder":
        """
        Set a query parameter directly.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Self for chaining
        """
        self._parameters[name] = value
        return self

    def build(self) -> tuple[str, Dict[str, Any]]:
        """
        Build the final Cypher query.

        Returns:
            Tuple of (query_string, parameters)
        """
        parts = []
        params = dict(self._parameters)

        # MATCH clauses
        if self._match_patterns:
            parts.append("MATCH " + ", ".join(self._match_patterns))

        # OPTIONAL MATCH clauses
        for pattern in self._optional_match_patterns:
            parts.append(f"OPTIONAL MATCH {pattern}")

        # WHERE clause
        if self._where_conditions:
            where_parts = []
            for cond in self._where_conditions:
                if isinstance(cond, _RawCondition):
                    where_parts.append(cond.cypher)
                    params.update(cond.params)
                else:
                    cypher, cond_params = cond.to_cypher(self._current_alias)
                    where_parts.append(cypher)
                    params.update(cond_params)
            parts.append("WHERE " + " AND ".join(where_parts))

        # WITH clauses
        for with_clause in self._with_clauses:
            parts.append(f"WITH {with_clause}")

        # CREATE clauses
        if self._create_patterns:
            parts.append("CREATE " + ", ".join(self._create_patterns))

        # SET clauses
        if self._set_clauses:
            parts.append("SET " + ", ".join(self._set_clauses))

        # DELETE clause
        if self._delete_items:
            delete_keyword = "DETACH DELETE" if self._detach_delete else "DELETE"
            parts.append(f"{delete_keyword} " + ", ".join(self._delete_items))

        # RETURN clause
        if self._return_items:
            parts.append("RETURN " + ", ".join(self._return_items))

        # ORDER BY clause
        if self._order_by:
            order_parts = []
            for criteria in self._order_by:
                alias = getattr(criteria, "_alias", self._current_alias)
                order_parts.append(criteria.to_cypher(alias))
            parts.append("ORDER BY " + ", ".join(order_parts))

        # SKIP clause
        if self._skip is not None:
            parts.append(f"SKIP {self._skip}")

        # LIMIT clause
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")

        query = "\n".join(parts)
        return query, params

    def __str__(self) -> str:
        """Return the query string."""
        query, _ = self.build()
        return query


@dataclass
class _RawCondition:
    """Internal class for raw Cypher conditions."""

    cypher: str
    params: Dict[str, Any]

    def to_cypher(self, alias: str = "n") -> tuple[str, Dict[str, Any]]:
        """Return raw cypher and params."""
        return self.cypher, self.params


# Convenience functions for common query patterns

def match_entity_by_id(entity_id: str, alias: str = "e") -> CypherQueryBuilder:
    """Build a query to match an entity by ID."""
    return (
        CypherQueryBuilder()
        .match("Entity", alias=alias)
        .where("id", ComparisonOperator.EQUALS, entity_id, param_name="entity_id")
        .return_node(alias)
    )


def match_entities_by_type(
    entity_type: str,
    limit: int = 100,
    alias: str = "e",
) -> CypherQueryBuilder:
    """Build a query to match entities by type."""
    return (
        CypherQueryBuilder()
        .match("Entity", alias=alias)
        .where("entity_type", ComparisonOperator.EQUALS, entity_type, param_name="entity_type")
        .return_node(alias)
        .order_by("created_at", SortDirection.DESC, alias)
        .limit(limit)
    )


def match_neighbors(
    entity_id: str,
    depth: int = 1,
    relationship_types: Optional[List[str]] = None,
) -> CypherQueryBuilder:
    """Build a query to find neighboring entities."""
    builder = CypherQueryBuilder().match("Entity", alias="start", properties={"id": entity_id})

    if relationship_types:
        # Multiple relationship types need special handling
        rel_pattern = "|".join(relationship_types)
        builder.match_relationship(
            "start",
            "neighbor:Entity",
            rel_type=rel_pattern,
            direction="both",
            min_hops=1,
            max_hops=depth,
        )
    else:
        builder.match_relationship(
            "start",
            "neighbor:Entity",
            direction="both",
            min_hops=1,
            max_hops=depth,
        )

    return builder.where_raw("neighbor.id <> $entity_id", {"entity_id": entity_id}).return_distinct("neighbor")


def search_entities_by_text(
    search_term: str,
    entity_types: Optional[List[str]] = None,
    limit: int = 100,
) -> CypherQueryBuilder:
    """Build a query to search entities by name or description."""
    builder = (
        CypherQueryBuilder()
        .match("Entity", alias="e")
        .where_group(
            [
                ("name", ComparisonOperator.CONTAINS, search_term),
                ("description", ComparisonOperator.CONTAINS, search_term),
            ],
            LogicalOperator.OR,
        )
    )

    if entity_types:
        builder.where("entity_type", ComparisonOperator.IN, entity_types, param_name="entity_types")

    return builder.return_node("e").order_by("name", alias="e").limit(limit)
