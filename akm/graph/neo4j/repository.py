"""Neo4j repository for CRUD operations abstraction."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import UUID

from neo4j import Session

from akm.core.exceptions import EntityNotFoundError
from akm.core.models import Entity, Relationship

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Neo4jRepository:
    """
    Repository pattern implementation for Neo4j operations.

    Provides a clean abstraction over Neo4j CRUD operations with:
    - Type-safe entity operations
    - Batch operations for performance
    - Transaction management
    - Query building helpers
    """

    def __init__(self, session_factory) -> None:
        """
        Initialize the repository.

        Args:
            session_factory: Callable that returns a Neo4j session
        """
        self._session_factory = session_factory

    def _get_session(self) -> Session:
        """Get a database session."""
        return self._session_factory()

    # Entity CRUD Operations

    def create_entity(self, entity: Entity) -> Entity:
        """
        Create a new entity in Neo4j.

        Args:
            entity: The entity to create

        Returns:
            The created entity
        """
        with self._get_session() as session:
            query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                entity_type: $entity_type,
                description: $description,
                properties: $properties,
                confidence: $confidence,
                domain_type: $domain_type,
                source_document_id: $source_document_id,
                created_at: $created_at,
                updated_at: $updated_at,
                metadata: $metadata
            })
            RETURN e
            """
            params = self._entity_to_params(entity)
            result = session.run(query, params)
            result.consume()
            logger.debug(f"Created entity: {entity.id}")
            return entity

    def create_entities_batch(self, entities: List[Entity]) -> List[Entity]:
        """
        Create multiple entities in a single transaction.

        Args:
            entities: List of entities to create

        Returns:
            List of created entities
        """
        if not entities:
            return []

        with self._get_session() as session:
            query = """
            UNWIND $entities AS e
            CREATE (n:Entity {
                id: e.id,
                name: e.name,
                entity_type: e.entity_type,
                description: e.description,
                properties: e.properties,
                confidence: e.confidence,
                domain_type: e.domain_type,
                source_document_id: e.source_document_id,
                created_at: e.created_at,
                updated_at: e.updated_at,
                metadata: e.metadata
            })
            RETURN n
            """
            params_list = [self._entity_to_params(e) for e in entities]
            result = session.run(query, {"entities": params_list})
            result.consume()
            logger.info(f"Batch created {len(entities)} entities")
            return entities

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: The entity ID

        Returns:
            The entity if found, None otherwise
        """
        with self._get_session() as session:
            query = """
            MATCH (e:Entity {id: $id})
            RETURN e
            """
            result = session.run(query, {"id": entity_id})
            record = result.single()
            if record:
                return self._node_to_entity(record["e"])
            return None

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """
        Retrieve an entity by name.

        Args:
            name: The entity name

        Returns:
            The entity if found, None otherwise
        """
        with self._get_session() as session:
            query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            LIMIT 1
            """
            result = session.run(query, {"name": name})
            record = result.single()
            if record:
                return self._node_to_entity(record["e"])
            return None

    def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entity]:
        """
        Get entities by type with pagination.

        Args:
            entity_type: The entity type to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of entities
        """
        with self._get_session() as session:
            query = """
            MATCH (e:Entity {entity_type: $entity_type})
            RETURN e
            ORDER BY e.created_at DESC
            SKIP $offset
            LIMIT $limit
            """
            result = session.run(
                query,
                {"entity_type": entity_type, "limit": limit, "offset": offset},
            )
            return [self._node_to_entity(record["e"]) for record in result]

    def get_entities_by_property(
        self,
        property_name: str,
        property_value: Any,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Get entities by a specific property value.

        Args:
            property_name: The property name to filter by
            property_value: The property value to match
            limit: Maximum number of results

        Returns:
            List of entities
        """
        with self._get_session() as session:
            query = f"""
            MATCH (e:Entity)
            WHERE e.properties.{property_name} = $value
            RETURN e
            LIMIT $limit
            """
            result = session.run(query, {"value": property_value, "limit": limit})
            return [self._node_to_entity(record["e"]) for record in result]

    def update_entity(self, entity: Entity) -> Entity:
        """
        Update an existing entity.

        Args:
            entity: The entity with updated values

        Returns:
            The updated entity
        """
        entity.updated_at = datetime.utcnow()
        with self._get_session() as session:
            query = """
            MATCH (e:Entity {id: $id})
            SET e.name = $name,
                e.entity_type = $entity_type,
                e.description = $description,
                e.properties = $properties,
                e.confidence = $confidence,
                e.domain_type = $domain_type,
                e.updated_at = $updated_at,
                e.metadata = $metadata
            RETURN e
            """
            params = self._entity_to_params(entity)
            result = session.run(query, params)
            record = result.single()
            if not record:
                raise EntityNotFoundError(str(entity.id))
            logger.debug(f"Updated entity: {entity.id}")
            return entity

    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.

        Args:
            entity_id: The entity ID

        Returns:
            True if deleted, False if not found
        """
        with self._get_session() as session:
            query = """
            MATCH (e:Entity {id: $id})
            DETACH DELETE e
            RETURN count(e) as deleted
            """
            result = session.run(query, {"id": entity_id})
            record = result.single()
            deleted = record["deleted"] > 0
            if deleted:
                logger.debug(f"Deleted entity: {entity_id}")
            return deleted

    def delete_entities_batch(self, entity_ids: List[str]) -> int:
        """
        Delete multiple entities in a single transaction.

        Args:
            entity_ids: List of entity IDs to delete

        Returns:
            Number of entities deleted
        """
        if not entity_ids:
            return 0

        with self._get_session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.id IN $ids
            DETACH DELETE e
            RETURN count(e) as deleted
            """
            result = session.run(query, {"ids": entity_ids})
            record = result.single()
            deleted = record["deleted"]
            logger.info(f"Batch deleted {deleted} entities")
            return deleted

    # Relationship CRUD Operations

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship: Relationship,
    ) -> Relationship:
        """
        Create a relationship between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship: The relationship to create

        Returns:
            The created relationship
        """
        rel_type = self._get_rel_type_string(relationship.relationship_type)

        with self._get_session() as session:
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            CREATE (source)-[r:{rel_type} {{
                id: $id,
                properties: $properties,
                confidence: $confidence,
                bidirectional: $bidirectional,
                domain_type: $domain_type,
                created_at: $created_at,
                updated_at: $updated_at,
                metadata: $metadata
            }}]->(target)
            RETURN r
            """
            params = {
                "source_id": source_id,
                "target_id": target_id,
                **self._relationship_to_params(relationship),
            }
            result = session.run(query, params)
            result.consume()
            logger.debug(f"Created relationship: {relationship.id}")
            return relationship

    def create_relationships_batch(
        self,
        relationships: List[tuple],
    ) -> List[Relationship]:
        """
        Create multiple relationships in a single transaction.

        Args:
            relationships: List of (source_id, target_id, relationship) tuples

        Returns:
            List of created relationships
        """
        if not relationships:
            return []

        created = []
        with self._get_session() as session:
            for source_id, target_id, rel in relationships:
                rel_type = self._get_rel_type_string(rel.relationship_type)
                query = f"""
                MATCH (source:Entity {{id: $source_id}})
                MATCH (target:Entity {{id: $target_id}})
                CREATE (source)-[r:{rel_type} {{
                    id: $id,
                    properties: $properties,
                    confidence: $confidence,
                    bidirectional: $bidirectional,
                    domain_type: $domain_type,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    metadata: $metadata
                }}]->(target)
                RETURN r
                """
                params = {
                    "source_id": source_id,
                    "target_id": target_id,
                    **self._relationship_to_params(rel),
                }
                result = session.run(query, params)
                result.consume()
                created.append(rel)

        logger.info(f"Batch created {len(created)} relationships")
        return created

    def get_relationship_by_id(self, relationship_id: str) -> Optional[Relationship]:
        """
        Retrieve a relationship by ID.

        Args:
            relationship_id: The relationship ID

        Returns:
            The relationship if found, None otherwise
        """
        with self._get_session() as session:
            query = """
            MATCH (source)-[r {id: $id}]->(target)
            RETURN r, source.id as source_id, target.id as target_id, type(r) as rel_type
            """
            result = session.run(query, {"id": relationship_id})
            record = result.single()
            if record:
                return self._rel_to_relationship(
                    record["r"],
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                )
            return None

    def get_relationships_for_entity(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
    ) -> List[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: The entity ID
            direction: "outgoing", "incoming", or "both"
            relationship_type: Optional type filter

        Returns:
            List of relationships
        """
        with self._get_session() as session:
            if direction == "outgoing":
                pattern = "(e:Entity {id: $entity_id})-[r]->(target)"
                return_clause = "r, e.id as source_id, target.id as target_id"
            elif direction == "incoming":
                pattern = "(source)-[r]->(e:Entity {id: $entity_id})"
                return_clause = "r, source.id as source_id, e.id as target_id"
            else:
                pattern = "(e:Entity {id: $entity_id})-[r]-(other)"
                return_clause = """
                r,
                CASE WHEN startNode(r) = e THEN e.id ELSE other.id END as source_id,
                CASE WHEN endNode(r) = e THEN e.id ELSE other.id END as target_id
                """

            type_filter = f":{relationship_type}" if relationship_type else ""
            query = f"""
            MATCH {pattern.replace('[r]', f'[r{type_filter}]')}
            RETURN {return_clause}, type(r) as rel_type
            """
            result = session.run(query, {"entity_id": entity_id})

            relationships = []
            for record in result:
                relationships.append(
                    self._rel_to_relationship(
                        record["r"],
                        record["source_id"],
                        record["target_id"],
                        record["rel_type"],
                    )
                )
            return relationships

    def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship by ID.

        Args:
            relationship_id: The relationship ID

        Returns:
            True if deleted, False if not found
        """
        with self._get_session() as session:
            query = """
            MATCH ()-[r {id: $id}]->()
            DELETE r
            RETURN count(r) as deleted
            """
            result = session.run(query, {"id": relationship_id})
            record = result.single()
            deleted = record["deleted"] > 0
            if deleted:
                logger.debug(f"Deleted relationship: {relationship_id}")
            return deleted

    # Search and Query Operations

    def search_entities(
        self,
        search_term: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Search entities by name or description.

        Args:
            search_term: The search term
            entity_types: Optional list of types to filter by
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        with self._get_session() as session:
            type_filter = ""
            if entity_types:
                type_filter = "AND e.entity_type IN $entity_types"

            query = f"""
            MATCH (e:Entity)
            WHERE (e.name CONTAINS $term OR e.description CONTAINS $term)
            {type_filter}
            RETURN e
            ORDER BY e.name
            LIMIT $limit
            """
            params = {"term": search_term, "limit": limit}
            if entity_types:
                params["entity_types"] = entity_types

            result = session.run(query, params)
            return [self._node_to_entity(record["e"]) for record in result]

    def count_entities(self, entity_type: Optional[str] = None) -> int:
        """
        Count entities, optionally filtered by type.

        Args:
            entity_type: Optional type to filter by

        Returns:
            Number of entities
        """
        with self._get_session() as session:
            if entity_type:
                query = """
                MATCH (e:Entity {entity_type: $entity_type})
                RETURN count(e) as count
                """
                result = session.run(query, {"entity_type": entity_type})
            else:
                query = """
                MATCH (e:Entity)
                RETURN count(e) as count
                """
                result = session.run(query)

            record = result.single()
            return record["count"]

    def count_relationships(self, relationship_type: Optional[str] = None) -> int:
        """
        Count relationships, optionally filtered by type.

        Args:
            relationship_type: Optional type to filter by

        Returns:
            Number of relationships
        """
        with self._get_session() as session:
            if relationship_type:
                query = f"""
                MATCH ()-[r:{relationship_type}]->()
                RETURN count(r) as count
                """
            else:
                query = """
                MATCH ()-[r]->()
                RETURN count(r) as count
                """
            result = session.run(query)
            record = result.single()
            return record["count"]

    # Helper Methods

    def _entity_to_params(self, entity: Entity) -> Dict[str, Any]:
        """Convert entity to query parameters."""
        entity_type = entity.entity_type
        if hasattr(entity_type, "value"):
            entity_type = entity_type.value

        return {
            "id": str(entity.id),
            "name": entity.name,
            "entity_type": entity_type,
            "description": entity.description,
            "properties": entity.properties,
            "confidence": entity.confidence,
            "domain_type": entity.domain_type,
            "source_document_id": (
                str(entity.source_document_id) if entity.source_document_id else None
            ),
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "metadata": entity.metadata,
        }

    def _relationship_to_params(self, relationship: Relationship) -> Dict[str, Any]:
        """Convert relationship to query parameters."""
        return {
            "id": str(relationship.id),
            "properties": relationship.properties,
            "confidence": relationship.confidence,
            "bidirectional": relationship.bidirectional,
            "domain_type": relationship.domain_type,
            "created_at": relationship.created_at.isoformat(),
            "updated_at": relationship.updated_at.isoformat(),
            "metadata": relationship.metadata,
        }

    def _get_rel_type_string(self, rel_type: Any) -> str:
        """Get relationship type as string."""
        if hasattr(rel_type, "value"):
            return rel_type.value
        return str(rel_type)

    def _node_to_entity(self, node: Any) -> Entity:
        """Convert a Neo4j node to an Entity."""
        props = dict(node)
        return Entity(
            id=UUID(props["id"]),
            name=props["name"],
            entity_type=props.get("entity_type", "generic"),
            description=props.get("description"),
            properties=props.get("properties", {}),
            confidence=props.get("confidence", 1.0),
            domain_type=props.get("domain_type"),
            source_document_id=(
                UUID(props["source_document_id"])
                if props.get("source_document_id")
                else None
            ),
            created_at=datetime.fromisoformat(props["created_at"]),
            updated_at=datetime.fromisoformat(props["updated_at"]),
            metadata=props.get("metadata", {}),
        )

    def _rel_to_relationship(
        self,
        rel: Any,
        source_id: str,
        target_id: str,
        rel_type: str,
    ) -> Relationship:
        """Convert a Neo4j relationship to a Relationship."""
        props = dict(rel)
        return Relationship(
            id=UUID(props["id"]),
            source_id=UUID(source_id),
            target_id=UUID(target_id),
            relationship_type=rel_type,
            properties=props.get("properties", {}),
            confidence=props.get("confidence", 1.0),
            bidirectional=props.get("bidirectional", False),
            domain_type=props.get("domain_type"),
            created_at=datetime.fromisoformat(props["created_at"]),
            updated_at=datetime.fromisoformat(props["updated_at"]),
            metadata=props.get("metadata", {}),
        )
