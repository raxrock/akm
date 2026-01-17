"""Domain adapters module."""

from typing import Optional

from akm.core.interfaces import DomainTransformer


def get_domain_transformer(domain_name: str) -> Optional[DomainTransformer]:
    """
    Get a domain transformer by name.

    Args:
        domain_name: Name of the domain adapter

    Returns:
        DomainTransformer instance or None if not found
    """
    domain_name = domain_name.lower()

    if domain_name == "software_engineering":
        from akm.adapters.software_engineering import SoftwareEngineeringTransformer

        return SoftwareEngineeringTransformer()

    elif domain_name == "generic":
        from akm.domain.transformer import BaseDomainTransformer

        return BaseDomainTransformer()

    return None


def list_available_domains() -> list[str]:
    """List available domain adapters."""
    return ["generic", "software_engineering"]


__all__ = ["get_domain_transformer", "list_available_domains"]
