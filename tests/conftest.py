"""Shared test fixtures for AKM framework."""

import pytest

from akm import AKM, AKMConfig, Entity, Link, Relationship
from akm.core.config import GraphConfig, LinkConfig, MemoryGraphConfig


@pytest.fixture
def test_config() -> AKMConfig:
    """Create test configuration with in-memory backends."""
    return AKMConfig(
        project_name="test_project",
        data_dir="./.akm_test",
        graph=GraphConfig(
            backend="memory",
            memory=MemoryGraphConfig(persist_path=None),
        ),
    )


@pytest.fixture
def akm_client(test_config: AKMConfig) -> AKM:
    """Create test AKM client with in-memory backend."""
    client = AKM(config=test_config)
    client.connect()
    yield client
    client.disconnect()


@pytest.fixture
def sample_entities() -> list[Entity]:
    """Create sample entities for testing."""
    return [
        Entity(name="UserService", entity_type="Class", description="Handles user operations"),
        Entity(name="AuthManager", entity_type="Class", description="Manages authentication"),
        Entity(name="Database", entity_type="concept", description="Data storage"),
    ]


@pytest.fixture
def link_config() -> LinkConfig:
    """Create link configuration for testing."""
    return LinkConfig(
        initial_soft_link_weight=0.5,
        promotion_threshold=0.8,
        demotion_threshold=0.2,
    )
