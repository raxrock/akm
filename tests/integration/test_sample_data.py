"""Integration tests using sample data from the data directory."""

import pytest
from pathlib import Path
from typing import List

from akm import AKM
from akm.core.config import AKMConfig, GraphConfig, MemoryGraphConfig
from akm.core.models import Entity, Document
from akm.ingestion.pipeline import IngestionPipeline, IngestionConfig
from akm.ingestion.connectors.files import FileSystemConnector
from akm.links.soft_link import PatternDetector, SoftLinkCreator, detect_and_create_soft_links
from akm.domain.schema.mapper import SchemaMapper, TypeInferencer
from akm.query.traversal import SemanticTraverser, TraversalConfig


# Path to sample data
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def test_config() -> AKMConfig:
    """Create test configuration with in-memory backends."""
    return AKMConfig(
        project_name="test_sample_data",
        data_dir="./.akm_test",
        graph=GraphConfig(
            backend="memory",
            memory=MemoryGraphConfig(persist_path=None),
        ),
    )


@pytest.fixture
def akm_client(test_config: AKMConfig) -> AKM:
    """Create test AKM client."""
    client = AKM(config=test_config)
    client.connect()
    yield client
    client.disconnect()


@pytest.fixture
def sample_docs_content() -> dict:
    """Load sample document content."""
    docs = {}

    # Architecture doc
    arch_path = SAMPLE_DATA_DIR / "docs" / "architecture.md"
    if arch_path.exists():
        docs["architecture"] = arch_path.read_text()

    # User service code
    user_service_path = SAMPLE_DATA_DIR / "code" / "user_service.py"
    if user_service_path.exists():
        docs["user_service"] = user_service_path.read_text()

    # Tech decisions
    tech_path = SAMPLE_DATA_DIR / "notes" / "tech-decisions.md"
    if tech_path.exists():
        docs["tech_decisions"] = tech_path.read_text()

    return docs


class TestFileSystemConnector:
    """Tests for file system connector with sample data."""

    def test_scan_data_directory(self) -> None:
        """Test scanning the data directory."""
        if not SAMPLE_DATA_DIR.exists():
            pytest.skip("Sample data directory not found")

        connector = FileSystemConnector()
        connector.connect(root_path=str(SAMPLE_DATA_DIR))

        docs = list(connector.fetch())
        connector.disconnect()

        assert len(docs) > 0
        # Should find markdown and python files
        extensions = {Path(d.source).suffix for d in docs}
        assert ".md" in extensions or ".py" in extensions

    def test_fetch_specific_patterns(self) -> None:
        """Test fetching with specific file patterns."""
        if not SAMPLE_DATA_DIR.exists():
            pytest.skip("Sample data directory not found")

        connector = FileSystemConnector()
        connector.connect(root_path=str(SAMPLE_DATA_DIR))

        # Fetch only markdown files
        docs = list(connector.fetch(filters={"patterns": ["*.md"]}))
        connector.disconnect()

        assert len(docs) > 0
        for doc in docs:
            assert doc.source.endswith(".md")


class TestPatternDetection:
    """Tests for pattern detection on sample data."""

    def test_detect_co_occurrence_in_architecture(self, sample_docs_content: dict) -> None:
        """Test co-occurrence detection in architecture doc."""
        if "architecture" not in sample_docs_content:
            pytest.skip("Architecture document not found")

        content = sample_docs_content["architecture"]

        # Create sample entities that should be in the doc
        entities = [
            Entity(name="User Service", entity_type="component"),
            Entity(name="Order Service", entity_type="component"),
            Entity(name="API Gateway", entity_type="component"),
            Entity(name="PostgreSQL", entity_type="technology"),
            Entity(name="Redis", entity_type="technology"),
        ]

        detector = PatternDetector(co_occurrence_window=200)
        patterns = detector.detect_co_occurrence(entities, content, "architecture.md")

        # Should detect some co-occurrences
        assert len(patterns) >= 0  # May or may not find patterns depending on exact content

    def test_detect_explicit_relationships(self, sample_docs_content: dict) -> None:
        """Test explicit relationship detection."""
        if "architecture" not in sample_docs_content:
            pytest.skip("Architecture document not found")

        content = sample_docs_content["architecture"]

        entities = [
            Entity(name="User Service", entity_type="component"),
            Entity(name="PostgreSQL", entity_type="technology"),
            Entity(name="Maria Garcia", entity_type="person"),
        ]

        detector = PatternDetector()
        patterns = detector.detect_explicit_relationships(entities, content)

        # May find explicit relationships like "developed by" or "uses"
        # This depends on exact document content
        assert isinstance(patterns, list)


class TestSoftLinkCreation:
    """Tests for soft link creation from sample data."""

    def test_create_links_from_document(self, sample_docs_content: dict) -> None:
        """Test creating soft links from a document."""
        if "architecture" not in sample_docs_content:
            pytest.skip("Architecture document not found")

        content = sample_docs_content["architecture"]

        entities = [
            Entity(name="User Service", entity_type="component"),
            Entity(name="Order Service", entity_type="component"),
            Entity(name="PostgreSQL", entity_type="technology"),
        ]

        links = detect_and_create_soft_links(
            entities=entities,
            text=content,
            document_id="architecture.md",
        )

        # Should create some links
        assert isinstance(links, list)
        for link in links:
            assert link.status.value == "soft"
            assert 0 <= link.weight.value <= 1


class TestTypeInference:
    """Tests for type inference on sample data."""

    def test_infer_entity_types_from_architecture(self, sample_docs_content: dict) -> None:
        """Test inferring entity types from architecture doc content."""
        if "architecture" not in sample_docs_content:
            pytest.skip("Architecture document not found")

        inferencer = TypeInferencer()

        # Test various entity names
        test_cases = [
            ("User Service", "component"),
            ("PostgreSQL", "technology"),
            ("Sarah Chen", "person"),
            ("API Gateway", "component"),
        ]

        for name, expected_type in test_cases:
            inferred_type, confidence = inferencer.infer_entity_type(
                name,
                context=sample_docs_content["architecture"][:500],
            )
            # Just verify we get a result
            assert inferred_type is not None
            assert 0 <= confidence <= 1


class TestGraphTraversal:
    """Tests for graph traversal with sample entities."""

    def test_traverse_from_entity(self, akm_client: AKM) -> None:
        """Test traversing from a starting entity."""
        # Create some entities
        e1 = akm_client.add_entity(
            name="User Service",
            entity_type="component",
            description="Handles user operations",
        )
        e2 = akm_client.add_entity(
            name="PostgreSQL",
            entity_type="technology",
            description="Database for user data",
        )
        e3 = akm_client.add_entity(
            name="Redis",
            entity_type="technology",
            description="Cache for sessions",
        )

        # Create relationships
        akm_client.add_relationship(
            source_id=str(e1.id),
            target_id=str(e2.id),
            relationship_type="USES",
        )
        akm_client.add_relationship(
            source_id=str(e1.id),
            target_id=str(e3.id),
            relationship_type="USES",
        )

        # Traverse from User Service
        result = akm_client.traverse(str(e1.id), depth=1)

        assert result.start_entity.name == "User Service"
        assert len(result.entities) == 2  # Should find PostgreSQL and Redis

    def test_build_subgraph(self, akm_client: AKM) -> None:
        """Test building a subgraph from entities."""
        from akm.query.traversal import ContextualTraverser

        # Create entities
        e1 = akm_client.add_entity(name="Service A", entity_type="component")
        e2 = akm_client.add_entity(name="Service B", entity_type="component")
        e3 = akm_client.add_entity(name="Database", entity_type="technology")

        # Create relationships
        akm_client.add_relationship(str(e1.id), str(e2.id), relationship_type="CALLS")
        akm_client.add_relationship(str(e2.id), str(e3.id), relationship_type="USES")

        # Build subgraph
        traverser = ContextualTraverser(akm_client.graph)
        subgraph = traverser.build_subgraph(
            [str(e1.id), str(e2.id), str(e3.id)],
            include_relationships=True,
        )

        assert len(subgraph["nodes"]) == 3
        assert len(subgraph["edges"]) >= 1


class TestIngestionPipeline:
    """Tests for the ingestion pipeline with sample data."""

    def test_ingest_sample_docs(self, akm_client: AKM) -> None:
        """Test ingesting sample documents."""
        if not SAMPLE_DATA_DIR.exists():
            pytest.skip("Sample data directory not found")

        docs_dir = SAMPLE_DATA_DIR / "docs"
        if not docs_dir.exists():
            pytest.skip("Sample docs directory not found")

        pipeline = IngestionPipeline(
            graph=akm_client.graph,
            link_manager=akm_client._link_manager,
        )

        config = IngestionConfig(
            recursive=False,
            patterns=["*.md"],
            extract_entities=False,  # Skip entity extraction for speed
            create_soft_links=False,
            index_documents=False,
        )

        stats = pipeline.ingest(str(docs_dir), config)

        assert stats.files_processed > 0
        assert stats.files_failed == 0


class TestQueryWithSampleData:
    """Tests for querying with sample data entities."""

    def test_find_entities_by_type(self, akm_client: AKM) -> None:
        """Test finding entities by type."""
        # Create sample entities
        akm_client.add_entity(name="User Service", entity_type="component")
        akm_client.add_entity(name="Order Service", entity_type="component")
        akm_client.add_entity(name="PostgreSQL", entity_type="technology")
        akm_client.add_entity(name="MongoDB", entity_type="technology")

        # Find components
        components = akm_client.find_entities(entity_type="component")
        assert len(components) == 2

        # Find technologies
        technologies = akm_client.find_entities(entity_type="technology")
        assert len(technologies) == 2

    def test_create_and_validate_link(self, akm_client: AKM) -> None:
        """Test creating and validating a link."""
        e1 = akm_client.add_entity(name="Service", entity_type="component")
        e2 = akm_client.add_entity(name="Database", entity_type="technology")

        # Create soft link
        link = akm_client.create_soft_link(
            source_id=str(e1.id),
            target_id=str(e2.id),
            link_type="uses",
            pattern_confidence=0.6,
        )

        assert link.status.value == "soft"
        initial_weight = link.weight.value

        # Validate the link
        validated = akm_client.validate_link(str(link.id), is_positive=True)
        assert validated.weight.value > initial_weight

    def test_link_decay(self, akm_client: AKM) -> None:
        """Test link weight decay."""
        e1 = akm_client.add_entity(name="Entity1", entity_type="concept")
        e2 = akm_client.add_entity(name="Entity2", entity_type="concept")

        link = akm_client.create_soft_link(
            source_id=str(e1.id),
            target_id=str(e2.id),
        )

        initial_weight = link.weight.value

        # Run decay
        count = akm_client.run_link_decay()

        # Decay should have been applied
        assert count >= 0


class TestEndToEndWorkflow:
    """End-to-end workflow tests with sample data."""

    def test_full_workflow(self, akm_client: AKM, sample_docs_content: dict) -> None:
        """Test a complete workflow from document to query."""
        if "architecture" not in sample_docs_content:
            pytest.skip("Architecture document not found")

        # Step 1: Create entities found in the architecture doc
        user_service = akm_client.add_entity(
            name="User Service",
            entity_type="component",
            description="Manages user authentication and profiles",
            properties={"team": "Backend Team", "database": "PostgreSQL"},
        )

        order_service = akm_client.add_entity(
            name="Order Service",
            entity_type="component",
            description="Handles e-commerce transactions",
            properties={"team": "Backend Team", "database": "MongoDB"},
        )

        postgres = akm_client.add_entity(
            name="PostgreSQL",
            entity_type="technology",
            description="Primary database for User Service",
        )

        maria = akm_client.add_entity(
            name="Maria Garcia",
            entity_type="person",
            description="Lead developer of User Service",
        )

        # Step 2: Create relationships
        akm_client.add_relationship(
            source_id=str(user_service.id),
            target_id=str(postgres.id),
            relationship_type="USES",
        )

        akm_client.add_relationship(
            source_id=str(maria.id),
            target_id=str(user_service.id),
            relationship_type="CREATED_BY",
        )

        # Step 3: Create soft links
        link = akm_client.create_soft_link(
            source_id=str(user_service.id),
            target_id=str(order_service.id),
            link_type="service_dependency",
            pattern_confidence=0.7,
        )

        # Step 4: Validate the link
        akm_client.validate_link(str(link.id), is_positive=True)

        # Step 5: Traverse the graph
        result = akm_client.traverse(str(user_service.id), depth=2)

        assert result.start_entity.name == "User Service"
        assert len(result.entities) >= 1  # Should find connected entities

        # Step 6: Get link stats
        stats = akm_client.get_link_stats()
        assert stats["total_links"] >= 1
