"""Base connector interface for data ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ConnectorResult:
    """Result from a connector scan."""

    source_path: str
    content: str
    content_type: str  # e.g., "text/plain", "text/markdown", "application/json"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    size_bytes: int = 0
    checksum: Optional[str] = None

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class BaseConnector(ABC):
    """
    Abstract base class for data connectors.

    Connectors are responsible for reading data from various sources
    (file systems, APIs, databases, etc.) and yielding standardized
    ConnectorResult objects for ingestion.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the connector.

        Args:
            name: Connector name/identifier
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    def scan(
        self,
        path: str,
        recursive: bool = True,
        patterns: Optional[List[str]] = None,
    ) -> Iterator[ConnectorResult]:
        """
        Scan the data source and yield results.

        Args:
            path: Path or identifier for what to scan
            recursive: Whether to scan recursively
            patterns: Optional glob patterns to filter by

        Yields:
            ConnectorResult objects for each discovered item
        """
        pass

    @abstractmethod
    def read(self, path: str) -> ConnectorResult:
        """
        Read a single item from the data source.

        Args:
            path: Path or identifier of the item to read

        Returns:
            ConnectorResult with the item content
        """
        pass

    def is_supported(self, path: str) -> bool:
        """
        Check if a path/identifier is supported by this connector.

        Args:
            path: Path or identifier to check

        Returns:
            True if supported, False otherwise
        """
        return True

    def get_content_type(self, path: str) -> str:
        """
        Determine content type from path.

        Args:
            path: Path to determine content type for

        Returns:
            MIME type string
        """
        ext = Path(path).suffix.lower()
        content_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".py": "text/x-python",
            ".js": "application/javascript",
            ".ts": "application/typescript",
            ".json": "application/json",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
            ".xml": "application/xml",
            ".html": "text/html",
            ".css": "text/css",
            ".java": "text/x-java",
            ".go": "text/x-go",
            ".rs": "text/x-rust",
            ".rb": "text/x-ruby",
            ".php": "text/x-php",
            ".c": "text/x-c",
            ".cpp": "text/x-c++",
            ".h": "text/x-c",
            ".hpp": "text/x-c++",
            ".sql": "application/sql",
            ".sh": "application/x-sh",
            ".bash": "application/x-sh",
        }
        return content_types.get(ext, "application/octet-stream")


__all__ = ["BaseConnector", "ConnectorResult"]
