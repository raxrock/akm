"""Tests for core data models."""

import pytest
from uuid import UUID

from akm.core.models import (
    Entity,
    EntityType,
    Link,
    LinkStatus,
    LinkWeight,
    Relationship,
    RelationshipType,
)


class TestEntity:
    """Test Entity model."""

    def test_create_entity(self) -> None:
        """Test basic entity creation."""
        entity = Entity(name="TestEntity", entity_type="concept")
        assert entity.name == "TestEntity"
        assert entity.entity_type == "concept"
        assert isinstance(entity.id, UUID)

    def test_entity_with_properties(self) -> None:
        """Test entity with custom properties."""
        entity = Entity(
            name="Service",
            entity_type=EntityType.CONCEPT,
            properties={"language": "python", "version": "3.10"},
        )
        assert entity.properties["language"] == "python"
        assert entity.properties["version"] == "3.10"

    def test_entity_confidence_validation(self) -> None:
        """Test confidence is clamped to valid range."""
        entity = Entity(name="Test", confidence=1.5)
        assert entity.confidence <= 1.0


class TestRelationship:
    """Test Relationship model."""

    def test_create_relationship(self) -> None:
        """Test basic relationship creation."""
        rel = Relationship(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            relationship_type=RelationshipType.DEPENDS_ON,
        )
        assert rel.relationship_type == RelationshipType.DEPENDS_ON
        assert not rel.bidirectional

    def test_bidirectional_relationship(self) -> None:
        """Test bidirectional relationship."""
        rel = Relationship(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            bidirectional=True,
        )
        assert rel.bidirectional is True


class TestLinkWeight:
    """Test LinkWeight model."""

    def test_default_weight(self) -> None:
        """Test default weight values."""
        weight = LinkWeight()
        assert weight.value == 0.5
        assert weight.decay_rate == 0.01
        assert weight.validation_count == 0

    def test_apply_positive_validation(self) -> None:
        """Test positive validation increases weight."""
        weight = LinkWeight(value=0.5)
        new_value = weight.apply_validation(is_positive=True, strength=0.1)
        assert new_value == 0.6
        assert weight.positive_validations == 1

    def test_apply_negative_validation(self) -> None:
        """Test negative validation decreases weight."""
        weight = LinkWeight(value=0.5)
        new_value = weight.apply_validation(is_positive=False, strength=0.1)
        assert new_value == 0.4
        assert weight.negative_validations == 1

    def test_weight_capped_at_one(self) -> None:
        """Test weight doesn't exceed 1.0."""
        weight = LinkWeight(value=0.95)
        weight.apply_validation(is_positive=True, strength=0.2)
        assert weight.value == 1.0

    def test_weight_capped_at_zero(self) -> None:
        """Test weight doesn't go below 0.0."""
        weight = LinkWeight(value=0.05)
        weight.apply_validation(is_positive=False, strength=0.2)
        assert weight.value == 0.0

    def test_should_promote(self) -> None:
        """Test promotion threshold check."""
        weight = LinkWeight(value=0.9, promotion_threshold=0.8)
        assert weight.should_promote() is True

        weight.value = 0.7
        assert weight.should_promote() is False

    def test_should_demote(self) -> None:
        """Test demotion threshold check."""
        weight = LinkWeight(value=0.1, demotion_threshold=0.2)
        assert weight.should_demote() is True

        weight.value = 0.3
        assert weight.should_demote() is False


class TestLink:
    """Test Link model."""

    def test_create_soft_link(self) -> None:
        """Test creating a soft link."""
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            link_type="co_occurrence",
        )
        assert link.status == LinkStatus.SOFT
        assert link.weight.value == 0.5

    def test_increment_co_occurrence(self) -> None:
        """Test co-occurrence increments boost weight."""
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
        )
        initial_weight = link.weight.value
        link.increment_co_occurrence()
        assert link.co_occurrence_count == 1
        assert link.weight.value > initial_weight

    def test_link_promotion_check(self) -> None:
        """Test link promotion threshold check."""
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            weight=LinkWeight(value=0.9, promotion_threshold=0.8),
        )
        assert link.should_promote() is True

    def test_link_demotion_check(self) -> None:
        """Test link demotion threshold check."""
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            weight=LinkWeight(value=0.1, demotion_threshold=0.2),
        )
        assert link.should_demote() is True
