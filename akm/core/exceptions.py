"""Custom exceptions for the AKM framework."""

from __future__ import annotations

from typing import Any, Optional


class AKMError(Exception):
    """Base exception for all AKM errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(AKMError):
    """Raised when there's a configuration error."""

    pass


class ConnectionError(AKMError):
    """Raised when a connection to a backend fails."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.backend = backend
        super().__init__(message, details)


class EntityNotFoundError(AKMError):
    """Raised when an entity is not found."""

    def __init__(
        self,
        entity_id: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.entity_id = entity_id
        msg = message or f"Entity not found: {entity_id}"
        super().__init__(msg, details)


class RelationshipNotFoundError(AKMError):
    """Raised when a relationship is not found."""

    def __init__(
        self,
        relationship_id: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.relationship_id = relationship_id
        msg = message or f"Relationship not found: {relationship_id}"
        super().__init__(msg, details)


class LinkNotFoundError(AKMError):
    """Raised when a link is not found."""

    def __init__(
        self,
        link_id: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.link_id = link_id
        msg = message or f"Link not found: {link_id}"
        super().__init__(msg, details)


class ValidationError(AKMError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.field = field
        self.value = value
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(message, details)


class SchemaError(AKMError):
    """Raised when there's a schema-related error."""

    pass


class IngestionError(AKMError):
    """Raised when data ingestion fails."""

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.source = source
        details = details or {}
        if source:
            details["source"] = source
        super().__init__(message, details)


class QueryError(AKMError):
    """Raised when a query fails."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.query = query
        details = details or {}
        if query:
            details["query"] = query
        super().__init__(message, details)


class EmbeddingError(AKMError):
    """Raised when embedding generation fails."""

    pass


class GNNError(AKMError):
    """Raised when GNN operations fail."""

    pass


class PluginError(AKMError):
    """Raised when plugin operations fail."""

    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.plugin_name = plugin_name
        details = details or {}
        if plugin_name:
            details["plugin_name"] = plugin_name
        super().__init__(message, details)


class BackendNotAvailableError(AKMError):
    """Raised when a required backend is not available."""

    def __init__(
        self,
        backend_type: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.backend_type = backend_type
        msg = message or f"Backend not available: {backend_type}"
        super().__init__(msg, details)
