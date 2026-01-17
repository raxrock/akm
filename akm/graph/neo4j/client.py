"""Neo4j graph backend implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from neo4j import GraphDatabase, Session
from neo4j.exceptions import Neo4jError

from akm.core.config import Neo4jConfig
from akm.core.exceptions import ConnectionError, EntityNotFoundError
from akm.core.interfaces import GraphBackend
from akm.core.models import Entity, Relationship, TraversalResult

logger = logging.getLogger(__name__)


class Neo4jBackend(GraphBackend[Entity, Relationship]):
    """Neo4j implementation of the graph backend."""

    def __init__(self, config: Neo4jConfig) -> None:
        """
        Initialize Neo4j backend.

        Args:
            config: Neo4j configuration
        """
        self._config = config
        self._driver = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self._config.uri,
                auth=(self._config.username, self._config.password),
                max_connection_pool_size=self._config.max_connection_pool_size,
                connection_timeout=self._config.connection_timeout,
                encrypted=self._config.encrypted,
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self._config.uri}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j: {e}",
                backend="neo4j",
                details={"uri": self._config.uri},
            )

    def disconnect(self) -> None:
        """Close connection to Neo4j database."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False
            logger.info("Disconnected from Neo4j")

    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected and self._driver is not None

    def _get_session(self) -> Session:
        """Get a database session."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Neo4j", backend="neo4j")
        return self._driver.session(database=self._config.database)

    def create_entity(self, entity: Entity) -> Entity:
        """Create a new entity in Neo4j."""
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
            params = {
                "id": str(entity.id),
                "name": entity.name,
                "entity_type": (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else entity.entity_type
                ),
                "description": entity.description,
                "properties": entity.properties,
                "confidence": entity.confidence,
                "domain_type": entity.domain_type,
                "source_document_id": (
                    str(entity.source_document_id)
                    if entity.source_document_id
                    else None
                ),
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "metadata": entity.metadata,
            }
            result = session.run(query, params)
            result.consume()
            logger.debug(f"Created entity: {entity.id}")
            return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
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

    def update_entity(self, entity: Entity) -> Entity:
        """Update an existing entity."""
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
            params = {
                "id": str(entity.id),
                "name": entity.name,
                "entity_type": (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else entity.entity_type
                ),
                "description": entity.description,
                "properties": entity.properties,
                "confidence": entity.confidence,
                "domain_type": entity.domain_type,
                "updated_at": entity.updated_at.isoformat(),
                "metadata": entity.metadata,
            }
            result = session.run(query, params)
            record = result.single()
            if not record:
                raise EntityNotFoundError(str(entity.id))
            return entity

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID."""
        with self._get_session() as session:
            query = """
            MATCH (e:Entity {id: $id})
            DETACH DELETE e
            RETURN count(e) as deleted
            """
            result = session.run(query, {"id": entity_id})
            record = result.single()
            return record["deleted"] > 0

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Find entities matching criteria."""
        with self._get_session() as session:
            conditions = []
            params: Dict[str, Any] = {"limit": limit}

            if entity_type:
                conditions.append("e.entity_type = $entity_type")
                params["entity_type"] = entity_type

            if properties:
                for key, value in properties.items():
                    param_name = f"prop_{key}"
                    conditions.append(f"e.properties.{key} = ${param_name}")
                    params[param_name] = value

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            query = f"""
            MATCH (e:Entity)
            {where_clause}
            RETURN e
            LIMIT $limit
            """
            result = session.run(query, params)
            return [self._node_to_entity(record["e"]) for record in result]

    def create_relationship(
        self, source_id: str, target_id: str, relationship: Relationship
    ) -> Relationship:
        """Create a relationship between two entities."""
        rel_type = (
            relationship.relationship_type.value
            if hasattr(relationship.relationship_type, "value")
            else relationship.relationship_type
        )

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
                "id": str(relationship.id),
                "properties": relationship.properties,
                "confidence": relationship.confidence,
                "bidirectional": relationship.bidirectional,
                "domain_type": relationship.domain_type,
                "created_at": relationship.created_at.isoformat(),
                "updated_at": relationship.updated_at.isoformat(),
                "metadata": relationship.metadata,
            }
            result = session.run(query, params)
            result.consume()
            logger.debug(f"Created relationship: {relationship.id}")
            return relationship

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve a relationship by ID."""
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

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        with self._get_session() as session:
            if direction == "outgoing":
                pattern = "(e:Entity {id: $entity_id})-[r]->(target)"
                return_clause = "r, e.id as source_id, target.id as target_id"
            elif direction == "incoming":
                pattern = "(source)-[r]->(e:Entity {id: $entity_id})"
                return_clause = "r, source.id as source_id, e.id as target_id"
            else:  # both
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
        """Delete a relationship by ID."""
        with self._get_session() as session:
            query = """
            MATCH ()-[r {id: $id}]->()
            DELETE r
            RETURN count(r) as deleted
            """
            result = session.run(query, {"id": relationship_id})
            record = result.single()
            return record["deleted"] > 0

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """Execute a native Cypher query."""
        with self._get_session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def traverse(
        self,
        start_id: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
    ) -> TraversalResult:
        """Traverse the graph from a starting node."""
        start_entity = self.get_entity(start_id)
        if not start_entity:
            raise EntityNotFoundError(start_id)

        with self._get_session() as session:
            rel_filter = (
                "|".join(f":{rt}" for rt in relationship_types)
                if relationship_types
                else ""
            )
            if rel_filter:
                rel_pattern = f"[r{rel_filter}*1..{depth}]"
            else:
                rel_pattern = f"[r*1..{depth}]"

            query = f"""
            MATCH path = (start:Entity {{id: $start_id}})-{rel_pattern}-(end:Entity)
            WITH nodes(path) as nodes, relationships(path) as rels
            UNWIND nodes as n
            WITH collect(DISTINCT n) as all_nodes, rels
            UNWIND rels as r
            RETURN all_nodes, collect(DISTINCT r) as all_rels
            """
            result = session.run(query, {"start_id": start_id})
            record = result.single()

            entities = []
            relationships = []

            if record:
                for node in record["all_nodes"]:
                    entity = self._node_to_entity(node)
                    if str(entity.id) != start_id:
                        entities.append(entity)

                for rel in record["all_rels"]:
                    # Get source and target from relationship
                    pass  # Complex to extract from path

            return TraversalResult(
                start_entity=start_entity,
                entities=entities,
                relationships=relationships,
                depth_reached=depth,
            )

    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """Get neighboring entities."""
        with self._get_session() as session:
            rel_filter = (
                "|".join(f":{rt}" for rt in relationship_types)
                if relationship_types
                else ""
            )
            if rel_filter:
                rel_pattern = f"[{rel_filter}*1..{depth}]"
            else:
                rel_pattern = f"[*1..{depth}]"

            query = f"""
            MATCH (start:Entity {{id: $entity_id}})-{rel_pattern}-(neighbor:Entity)
            WHERE neighbor.id <> $entity_id
            RETURN DISTINCT neighbor
            """
            result = session.run(query, {"entity_id": entity_id})
            return [self._node_to_entity(record["neighbor"]) for record in result]

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
        self, rel: Any, source_id: str, target_id: str, rel_type: str
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
