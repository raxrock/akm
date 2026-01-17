"""Data connectors module."""

from akm.ingestion.connectors.base import BaseConnector, ConnectorResult
from akm.ingestion.connectors.files import FileSystemConnector

__all__ = ["BaseConnector", "ConnectorResult", "FileSystemConnector"]
