"""Integration tests for the AKM client."""

import pytest

from akm import AKM, AKMConfig, Entity, LinkStatus
from akm.core.config import GraphConfig, LinkConfig, MemoryGraphConfig


@pytest.fixture
def akm() -> AKM:
    """Create AKM client with in-memory backend."""
    config = AKMConfig(
        project_name="test_project",
        graph=GraphConfig(
            backend="memory",
            memory=MemoryGraphConfig(persist_path=None),
        ),
    )
    client = AKM(config=config)
    client.connect()
    yield client
    client.disconnect()


class TestEntityOperations:
    """Test entity CRUD operations."""

    def test_create_entity(self, akm: AKM) -> None:
        """Test creating an entity."""
        entity = akm.add_entity(
            name="TestService",
            entity_type="Class",
            description="A test service",
            properties={"language": "python"},
        )
        assert entity.name == "TestService"
        assert entity.entity_type == "Class"
        assert entity.properties["language"] == "python"

    def test_get_entity(self, akm: AKM) -> None:
        """Test retrieving an entity."""
        created = akm.add_entity(name="MyEntity", entity_type="concept")
        retrieved = akm.get_entity(str(created.id))
        assert retrieved is not None
        assert retrieved.name == "MyEntity"

    def test_find_entities(self, akm: AKM) -> None:
        """Test finding entities by type."""
        akm.add_entity(name="Entity1", entity_type="Class")
        akm.add_entity(name="Entity2", entity_type="Class")
        akm.add_entity(name="Entity3", entity_type="concept")

        classes = akm.find_entities(entity_type="Class")
        assert len(classes) == 2

    def test_delete_entity(self, akm: AKM) -> None:
        """Test deleting an entity."""
        entity = akm.add_entity(name="ToDelete", entity_type="concept")
        entity_id = str(entity.id)

        result = akm.delete_entity(entity_id)
        assert result is True

        retrieved = akm.get_entity(entity_id)
        assert retrieved is None


class TestRelationshipOperations:
    """Test relationship operations."""

    def test_create_relationship(self, akm: AKM) -> None:
        """Test creating a relationship."""
        e1 = akm.add_entity(name="Source", entity_type="Class")
        e2 = akm.add_entity(name="Target", entity_type="Class")

        rel = akm.add_relationship(
            source_id=str(e1.id),
            target_id=str(e2.id),
            relationship_type="DEPENDS_ON",
        )
        assert rel.relationship_type == "DEPENDS_ON"

    def test_get_relationships(self, akm: AKM) -> None:
        """Test getting relationships for an entity."""
        e1 = akm.add_entity(name="Source", entity_type="Class")
        e2 = akm.add_entity(name="Target", entity_type="Class")

        akm.add_relationship(
            source_id=str(e1.id),
            target_id=str(e2.id),
            relationship_type="DEPENDS_ON",
        )

        rels = akm.get_relationships(str(e1.id), direction="outgoing")
        assert len(rels) == 1
        assert rels[0].relationship_type == "DEPENDS_ON"


class TestLinkOperations:
    """Test adaptive link operations."""

    def test_create_soft_link(self, akm: AKM) -> None:
        """Test creating a soft link."""
        e1 = akm.add_entity(name="Entity1", entity_type="concept")
        e2 = akm.add_entity(name="Entity2", entity_type="concept")

        link = akm.create_soft_link(
            source_id=str(e1.id),
            target_id=str(e2.id),
            pattern_source="test",
            pattern_confidence=0.7,
        )

        assert link.status == LinkStatus.SOFT
        assert link.pattern_confidence == 0.7

    def test_validate_link_positive(self, akm: AKM) -> None:
        """Test positive link validation."""
        e1 = akm.add_entity(name="Entity1", entity_type="concept")
        e2 = akm.add_entity(name="Entity2", entity_type="concept")

        link = akm.create_soft_link(
            source_id=str(e1.id),
            target_id=str(e2.id),
        )
        initial_weight = link.weight.value

        updated = akm.validate_link(str(link.id), is_positive=True)
        assert updated.weight.value > initial_weight

    def test_validate_link_negative(self, akm: AKM) -> None:
        """Test negative link validation."""
        e1 = akm.add_entity(name="Entity1", entity_type="concept")
        e2 = akm.add_entity(name="Entity2", entity_type="concept")

        link = akm.create_soft_link(
            source_id=str(e1.id),
            target_id=str(e2.id),
        )
        initial_weight = link.weight.value

        updated = akm.validate_link(str(link.id), is_positive=False)
        assert updated.weight.value < initial_weight

    def test_get_links(self, akm: AKM) -> None:
        """Test getting links for an entity."""
        e1 = akm.add_entity(name="Entity1", entity_type="concept")
        e2 = akm.add_entity(name="Entity2", entity_type="concept")
        e3 = akm.add_entity(name="Entity3", entity_type="concept")

        akm.create_soft_link(source_id=str(e1.id), target_id=str(e2.id))
        akm.create_soft_link(source_id=str(e1.id), target_id=str(e3.id))

        links = akm.get_links(str(e1.id))
        assert len(links) == 2

    def test_link_co_occurrence_strengthens(self, akm: AKM) -> None:
        """Test that repeated co-occurrence strengthens links."""
        e1 = akm.add_entity(name="Entity1", entity_type="concept")
        e2 = akm.add_entity(name="Entity2", entity_type="concept")

        link1 = akm.create_soft_link(source_id=str(e1.id), target_id=str(e2.id))
        weight1 = link1.weight.value

        # Creating same link again should increment co-occurrence
        link2 = akm.create_soft_link(source_id=str(e1.id), target_id=str(e2.id))
        assert link2.co_occurrence_count > 1
        assert link2.weight.value > weight1


class TestTraversal:
    """Test graph traversal."""

    def test_traverse_graph(self, akm: AKM) -> None:
        """Test traversing from an entity."""
        e1 = akm.add_entity(name="Root", entity_type="concept")
        e2 = akm.add_entity(name="Child1", entity_type="concept")
        e3 = akm.add_entity(name="Child2", entity_type="concept")

        akm.add_relationship(str(e1.id), str(e2.id), "RELATED_TO")
        akm.add_relationship(str(e1.id), str(e3.id), "RELATED_TO")

        result = akm.traverse(str(e1.id), depth=1)
        assert result.start_entity.name == "Root"
        assert len(result.entities) == 2

    def test_get_neighbors(self, akm: AKM) -> None:
        """Test getting neighboring entities."""
        e1 = akm.add_entity(name="Center", entity_type="concept")
        e2 = akm.add_entity(name="Neighbor1", entity_type="concept")
        e3 = akm.add_entity(name="Neighbor2", entity_type="concept")

        akm.add_relationship(str(e1.id), str(e2.id), "RELATED_TO")
        akm.add_relationship(str(e1.id), str(e3.id), "RELATED_TO")

        neighbors = akm.get_neighbors(str(e1.id))
        assert len(neighbors) == 2


class TestLinkDecay:
    """Test link decay functionality."""

    def test_run_decay(self, akm: AKM) -> None:
        """Test running decay on links."""
        e1 = akm.add_entity(name="Entity1", entity_type="concept")
        e2 = akm.add_entity(name="Entity2", entity_type="concept")

        akm.create_soft_link(source_id=str(e1.id), target_id=str(e2.id))

        # Run decay (may not actually decay if not enough time passed)
        count = akm.run_link_decay()
        assert count >= 0  # Just verify it runs without error

    def test_link_stats(self, akm: AKM) -> None:
        """Test getting link statistics."""
        e1 = akm.add_entity(name="Entity1", entity_type="concept")
        e2 = akm.add_entity(name="Entity2", entity_type="concept")

        akm.create_soft_link(source_id=str(e1.id), target_id=str(e2.id))

        stats = akm.get_link_stats()
        assert stats["total_links"] == 1
        assert "average_weight" in stats
