"""Link validation through user interactions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from akm.core.config import LinkValidationConfig
from akm.core.models import Link, LinkStatus


def validate_link(
    link: Link,
    is_positive: bool,
    config: LinkValidationConfig,
    strength: Optional[float] = None,
    current_time: Optional[datetime] = None,
) -> Link:
    """
    Apply user validation to a link.

    When a user interacts with a link (e.g., clicking on a related suggestion,
    marking something as helpful), this strengthens or weakens the link.

    Args:
        link: The link to validate
        is_positive: True if the user found the link helpful
        config: Validation configuration
        strength: Optional override for validation strength
        current_time: Current time (defaults to UTC now)

    Returns:
        The link with updated weight and status
    """
    current_time = current_time or datetime.now(timezone.utc)

    # Determine validation strength
    if strength is None:
        strength = (
            config.positive_weight_boost
            if is_positive
            else config.negative_weight_penalty
        )

    # Apply validation to weight
    link.weight.apply_validation(is_positive, strength)

    # Update link status
    link = update_link_status(link, config)
    link.updated_at = current_time

    return link


def update_link_status(
    link: Link,
    config: LinkValidationConfig,
) -> Link:
    """
    Update link status based on current weight and validations.

    Args:
        link: The link to update
        config: Validation configuration

    Returns:
        The link with updated status
    """
    weight = link.weight

    # Auto-promote if weight exceeds threshold
    if weight.value >= weight.promotion_threshold:
        if link.status in (LinkStatus.SOFT, LinkStatus.VALIDATING, LinkStatus.DECAYING):
            link.status = LinkStatus.VALIDATED

    # Demote if weight drops below threshold
    elif weight.value <= weight.demotion_threshold:
        link.status = LinkStatus.ARCHIVED

    # Mark as validating if has validations but not yet promoted
    elif weight.validation_count > 0 and link.status == LinkStatus.SOFT:
        link.status = LinkStatus.VALIDATING

    # Check for auto-validation based on pattern confidence
    elif (
        link.pattern_confidence >= config.auto_validate_threshold
        and link.status == LinkStatus.SOFT
    ):
        link.status = LinkStatus.VALIDATED

    return link


def calculate_validation_score(link: Link) -> float:
    """
    Calculate a validation score for a link.

    The score considers:
    - Ratio of positive to negative validations
    - Total number of validations
    - Pattern confidence

    Args:
        link: The link to score

    Returns:
        Validation score between 0 and 1
    """
    weight = link.weight

    if weight.validation_count == 0:
        # No validations, use pattern confidence
        return link.pattern_confidence * 0.5

    # Calculate positive ratio
    positive_ratio = weight.positive_validations / weight.validation_count

    # Weight by number of validations (more validations = more confidence)
    validation_confidence = min(1.0, weight.validation_count / 10)

    # Combine with pattern confidence
    score = (
        positive_ratio * validation_confidence * 0.7
        + link.pattern_confidence * 0.3
    )

    return min(1.0, max(0.0, score))


def should_auto_validate(link: Link, config: LinkValidationConfig) -> bool:
    """
    Check if a link should be automatically validated.

    Auto-validation happens when:
    - Pattern confidence is very high
    - Co-occurrence is high
    - Semantic similarity is high

    Args:
        link: The link to check
        config: Validation configuration

    Returns:
        True if the link should be auto-validated
    """
    # High pattern confidence
    if link.pattern_confidence >= config.auto_validate_threshold:
        return True

    # High co-occurrence with good semantic similarity
    if (
        link.co_occurrence_count >= 5
        and link.semantic_similarity is not None
        and link.semantic_similarity >= 0.85
    ):
        return True

    # Strong graph-based evidence
    if link.common_neighbors >= 3:
        return True

    return False


def bulk_validate(
    links: list[Link],
    validations: list[tuple[str, bool]],  # (link_id, is_positive)
    config: LinkValidationConfig,
) -> dict[str, Link]:
    """
    Apply bulk validations to multiple links.

    Args:
        links: List of links to potentially validate
        validations: List of (link_id, is_positive) tuples
        config: Validation configuration

    Returns:
        Dictionary of link_id -> updated link
    """
    link_map = {str(link.id): link for link in links}
    results = {}

    for link_id, is_positive in validations:
        if link_id in link_map:
            link = link_map[link_id]
            updated = validate_link(link, is_positive, config)
            results[link_id] = updated

    return results
