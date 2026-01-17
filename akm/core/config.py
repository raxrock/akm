"""Configuration management for the AKM framework."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# =============================================================================
# Graph Configuration
# =============================================================================


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    encrypted: bool = False


class MemoryGraphConfig(BaseModel):
    """In-memory graph configuration for testing."""

    persist_path: Optional[str] = None


class GraphConfig(BaseModel):
    """Graph backend configuration."""

    backend: str = "neo4j"  # "neo4j" | "memory"
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    memory: MemoryGraphConfig = Field(default_factory=MemoryGraphConfig)


# =============================================================================
# Vector Configuration
# =============================================================================


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration."""

    persist_directory: str = "./.akm/chroma"
    collection_name: str = "akm_default"
    host: Optional[str] = None  # For client-server mode
    port: Optional[int] = None
    anonymized_telemetry: bool = False


class VectorConfig(BaseModel):
    """Vector backend configuration."""

    backend: str = "chromadb"  # "chromadb" | "none" | future backends
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50


# =============================================================================
# Embedding Configuration
# =============================================================================


class SentenceTransformersConfig(BaseModel):
    """Sentence Transformers configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    batch_size: int = 32


class OpenAIEmbeddingConfig(BaseModel):
    """OpenAI embedding configuration."""

    api_key: Optional[str] = None  # Can be set via env var OPENAI_API_KEY
    model: str = "text-embedding-3-small"
    batch_size: int = 100


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: str = "sentence_transformers"  # "sentence_transformers" | "openai"
    sentence_transformers: SentenceTransformersConfig = Field(
        default_factory=SentenceTransformersConfig
    )
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)


# =============================================================================
# LLM Configuration
# =============================================================================


class LangChainConfig(BaseModel):
    """LangChain LLM configuration."""

    provider: str = "openai"  # "openai" | "anthropic" | "ollama"
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    api_key: Optional[str] = None


class LlamaIndexConfig(BaseModel):
    """LlamaIndex configuration."""

    provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.0
    chunk_size: int = 1024
    chunk_overlap: int = 20


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    orchestrator: str = "langchain"  # "langchain" | "llamaindex"
    langchain: LangChainConfig = Field(default_factory=LangChainConfig)
    llamaindex: LlamaIndexConfig = Field(default_factory=LlamaIndexConfig)


# =============================================================================
# Link Lifecycle Configuration
# =============================================================================


class LinkDecayConfig(BaseModel):
    """Link decay configuration."""

    enabled: bool = True
    decay_rate: float = 0.01  # Per hour exponential decay rate
    decay_interval_hours: int = 24  # How often to run decay
    minimum_weight: float = 0.1  # Archive links below this


class LinkValidationConfig(BaseModel):
    """Link validation configuration."""

    positive_weight_boost: float = 0.1
    negative_weight_penalty: float = 0.15
    auto_validate_threshold: float = 0.9  # Auto-validate strong patterns


class LinkConfig(BaseModel):
    """Adaptive link lifecycle configuration."""

    initial_soft_link_weight: float = 0.5
    promotion_threshold: float = 0.8
    demotion_threshold: float = 0.2
    decay: LinkDecayConfig = Field(default_factory=LinkDecayConfig)
    validation: LinkValidationConfig = Field(default_factory=LinkValidationConfig)


# =============================================================================
# GNN Configuration
# =============================================================================


class GNNConfig(BaseModel):
    """Graph Neural Network configuration."""

    enabled: bool = False  # Disabled by default (requires torch)
    framework: str = "pyg"  # "pyg" (PyTorch Geometric) | "dgl"

    # Link prediction
    link_prediction_enabled: bool = True
    link_prediction_model: str = "GraphSAGE"
    link_prediction_threshold: float = 0.7

    # Community detection
    community_detection_enabled: bool = True
    community_detection_algorithm: str = "Louvain"

    # Training
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    learning_rate: float = 0.01
    epochs: int = 100


# =============================================================================
# Ingestion Configuration
# =============================================================================


class FileConnectorConfig(BaseModel):
    """File system connector configuration."""

    enabled: bool = True
    paths: List[str] = Field(default_factory=list)
    extensions: List[str] = Field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".py", ".js", ".ts"]
    )
    recursive: bool = True
    ignore_patterns: List[str] = Field(
        default_factory=lambda: ["*.pyc", "__pycache__", ".git", "node_modules"]
    )


class SlackConnectorConfig(BaseModel):
    """Slack connector configuration."""

    enabled: bool = False
    token: Optional[str] = None
    channels: List[str] = Field(default_factory=list)
    fetch_threads: bool = True


class IngestionConfig(BaseModel):
    """Ingestion pipeline configuration."""

    batch_size: int = 100
    parallel_workers: int = 4
    deduplication_enabled: bool = True
    file_connector: FileConnectorConfig = Field(default_factory=FileConnectorConfig)
    slack_connector: SlackConnectorConfig = Field(default_factory=SlackConnectorConfig)


# =============================================================================
# Domain Configuration
# =============================================================================


class DomainConfig(BaseModel):
    """Domain transformer configuration."""

    name: str = "generic"
    schema_path: Optional[str] = None
    custom_embedding_model: Optional[str] = None
    prompt_templates_path: Optional[str] = None


# =================================G============================================
# Plugin Configuration
# =============================================================================


class PluginConfig(BaseModel):
    """Plugin configuration."""

    enabled: bool = True
    auto_discover: bool = True
    plugin_paths: List[str] = Field(default_factory=list)
    disabled_plugins: List[str] = Field(default_factory=list)


# =============================================================================
# Main Configuration
# =============================================================================


class AKMConfig(BaseSettings):
    """Main AKM framework configuration."""

    # Core settings
    project_name: str = "akm_project"
    data_dir: str = "./.akm"
    log_level: LogLevel = LogLevel.INFO

    # Component configurations
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    links: LinkConfig = Field(default_factory=LinkConfig)
    gnn: GNNConfig = Field(default_factory=GNNConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)

    model_config = {
        "env_prefix": "AKM_",
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AKMConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_data_path(self, *parts: str) -> Path:
        """Get a path within the data directory."""
        path = Path(self.data_dir)
        for part in parts:
            path = path / part
        return path

    def ensure_data_dir(self) -> Path:
        """Ensure the data directory exists."""
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: Optional[Union[str, Path]] = None) -> AKMConfig:
    """
    Load configuration from file or create default.

    Args:
        path: Optional path to YAML config file

    Returns:
        AKMConfig instance
    """
    if path:
        return AKMConfig.from_yaml(path)
    return AKMConfig()
