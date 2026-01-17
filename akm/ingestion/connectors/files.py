"""File system connector for local file ingestion."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from akm.ingestion.connectors.base import BaseConnector, ConnectorResult

logger = logging.getLogger(__name__)


class FileSystemConnector(BaseConnector):
    """
    Connector for reading files from the local file system.

    Example usage:
        connector = FileSystemConnector()
        connector.connect()

        for result in connector.scan("./src", patterns=["*.py", "*.md"]):
            print(f"Found: {result.source_path}")

        connector.disconnect()
    """

    # Default patterns to exclude
    DEFAULT_EXCLUDE_PATTERNS = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "*.egg-info",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".DS_Store",
    ]

    # Default file patterns to include
    DEFAULT_INCLUDE_PATTERNS = [
        "*.py",
        "*.js",
        "*.ts",
        "*.jsx",
        "*.tsx",
        "*.java",
        "*.go",
        "*.rs",
        "*.rb",
        "*.php",
        "*.c",
        "*.cpp",
        "*.h",
        "*.hpp",
        "*.md",
        "*.txt",
        "*.json",
        "*.yaml",
        "*.yml",
        "*.xml",
        "*.sql",
        "*.sh",
    ]

    def __init__(
        self,
        name: str = "filesystem",
        config: Optional[Dict[str, Any]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size_mb: float = 10.0,
        encoding: str = "utf-8",
    ) -> None:
        """
        Initialize the file system connector.

        Args:
            name: Connector name
            config: Optional configuration
            exclude_patterns: Patterns to exclude (added to defaults)
            max_file_size_mb: Maximum file size in MB to process
            encoding: File encoding to use
        """
        super().__init__(name, config)
        self.exclude_patterns = self.DEFAULT_EXCLUDE_PATTERNS.copy()
        if exclude_patterns:
            self.exclude_patterns.extend(exclude_patterns)
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.encoding = encoding
        self._connected = False

    def connect(self) -> None:
        """Mark as connected (file system doesn't need real connection)."""
        self._connected = True
        logger.info("FileSystemConnector connected")

    def disconnect(self) -> None:
        """Mark as disconnected."""
        self._connected = False
        logger.info("FileSystemConnector disconnected")

    def scan(
        self,
        path: str,
        recursive: bool = True,
        patterns: Optional[List[str]] = None,
    ) -> Iterator[ConnectorResult]:
        """
        Scan a directory for files matching patterns.

        Args:
            path: Directory path to scan
            recursive: Whether to scan subdirectories
            patterns: Glob patterns to include (defaults to common code files)

        Yields:
            ConnectorResult for each matching file
        """
        if not self._connected:
            raise RuntimeError("Connector not connected. Call connect() first.")

        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not base_path.is_dir():
            # Single file
            if self._should_include(base_path, patterns):
                yield self.read(str(base_path))
            return

        include_patterns = patterns or self.DEFAULT_INCLUDE_PATTERNS

        # Walk directory
        if recursive:
            items = base_path.rglob("*")
        else:
            items = base_path.glob("*")

        for item in items:
            if not item.is_file():
                continue

            # Check exclusions first
            if self._should_exclude(item):
                continue

            # Check inclusions
            if not self._should_include(item, include_patterns):
                continue

            # Check file size
            try:
                if item.stat().st_size > self.max_file_size_bytes:
                    logger.warning(f"Skipping large file: {item}")
                    continue
            except OSError as e:
                logger.warning(f"Cannot stat file {item}: {e}")
                continue

            # Read and yield
            try:
                yield self.read(str(item))
            except Exception as e:
                logger.warning(f"Failed to read {item}: {e}")
                continue

    def read(self, path: str) -> ConnectorResult:
        """
        Read a single file.

        Args:
            path: File path to read

        Returns:
            ConnectorResult with file content and metadata
        """
        if not self._connected:
            raise RuntimeError("Connector not connected. Call connect() first.")

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Read content
        try:
            content = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            # Try binary read and decode with fallback
            content = file_path.read_bytes().decode(self.encoding, errors="replace")

        # Get file stats
        stat = file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        modified_at = datetime.fromtimestamp(stat.st_mtime)

        # Calculate checksum
        checksum = hashlib.md5(content.encode()).hexdigest()

        return ConnectorResult(
            source_path=str(file_path.absolute()),
            content=content,
            content_type=self.get_content_type(path),
            metadata={
                "filename": file_path.name,
                "extension": file_path.suffix,
                "relative_path": str(file_path),
                "encoding": self.encoding,
            },
            created_at=created_at,
            modified_at=modified_at,
            size_bytes=stat.st_size,
            checksum=checksum,
        )

    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded."""
        path_str = str(path)
        path_parts = path.parts

        for pattern in self.exclude_patterns:
            # Check if any path component matches
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
            # Check full path match
            if fnmatch.fnmatch(path_str, f"*{pattern}*"):
                return True

        return False

    def _should_include(self, path: Path, patterns: Optional[List[str]]) -> bool:
        """Check if a path should be included based on patterns."""
        if patterns is None:
            return True

        name = path.name
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return True

        return False


__all__ = ["FileSystemConnector"]
