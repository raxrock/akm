"""Adaptive link lifecycle module for the AKM framework."""

from akm.links.decay import (
    apply_decay_to_weight,
    calculate_decay_factor,
    calculate_half_life_hours,
    calculate_time_to_threshold,
    decay_link,
    decay_links_batch,
    should_decay,
)
from akm.links.manager import LinkManager
from akm.links.soft_link import (
    CoOccurrencePattern,
    PatternDetector,
    PatternMatch,
    SemanticPattern,
    SoftLinkCreator,
    StructuralPattern,
    compute_cosine_similarity,
    detect_and_create_soft_links,
)
from akm.links.validation import (
    bulk_validate,
    calculate_validation_score,
    should_auto_validate,
    update_link_status,
    validate_link,
)

__all__ = [
    # Manager
    "LinkManager",
    # Decay functions
    "apply_decay_to_weight",
    "calculate_decay_factor",
    "calculate_half_life_hours",
    "calculate_time_to_threshold",
    "decay_link",
    "decay_links_batch",
    "should_decay",
    # Validation functions
    "bulk_validate",
    "calculate_validation_score",
    "should_auto_validate",
    "update_link_status",
    "validate_link",
    # Soft link creation
    "PatternDetector",
    "SoftLinkCreator",
    "PatternMatch",
    "CoOccurrencePattern",
    "SemanticPattern",
    "StructuralPattern",
    "detect_and_create_soft_links",
    "compute_cosine_similarity",
]
