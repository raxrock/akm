"""Time-based decay functions for adaptive links."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional

from akm.core.config import LinkDecayConfig
from akm.core.models import Link, LinkStatus


def calculate_decay_factor(
    decay_rate: float,
    hours_elapsed: float,
) -> float:
    """
    Calculate the exponential decay factor.

    Uses the formula: decay_factor = exp(-decay_rate * hours)

    Args:
        decay_rate: The decay rate per hour (0.01 = 1% decay per hour)
        hours_elapsed: Number of hours elapsed since last decay

    Returns:
        Decay factor between 0 and 1
    """
    return math.exp(-decay_rate * hours_elapsed)


def apply_decay_to_weight(
    current_weight: float,
    decay_rate: float,
    hours_elapsed: float,
    minimum_weight: float = 0.0,
) -> float:
    """
    Apply exponential decay to a weight value.

    Args:
        current_weight: The current weight value
        decay_rate: The decay rate per hour
        hours_elapsed: Hours elapsed since last decay
        minimum_weight: Minimum weight to prevent complete decay

    Returns:
        New weight after decay
    """
    decay_factor = calculate_decay_factor(decay_rate, hours_elapsed)
    new_weight = current_weight * decay_factor
    return max(minimum_weight, new_weight)


def should_decay(
    link: Link,
    config: LinkDecayConfig,
    current_time: Optional[datetime] = None,
) -> bool:
    """
    Determine if a link should undergo decay.

    Args:
        link: The link to check
        config: Decay configuration
        current_time: Current time (defaults to UTC now)

    Returns:
        True if the link should decay
    """
    if not config.enabled:
        return False

    # Archived links don't decay further
    if link.status == LinkStatus.ARCHIVED:
        return False

    current_time = current_time or datetime.now(timezone.utc)
    last_decay = link.weight.last_decay_at

    # Handle timezone-naive datetimes
    if last_decay.tzinfo is None:
        last_decay = last_decay.replace(tzinfo=timezone.utc)

    hours_since_decay = (current_time - last_decay).total_seconds() / 3600
    return hours_since_decay >= config.decay_interval_hours


def decay_link(
    link: Link,
    config: LinkDecayConfig,
    current_time: Optional[datetime] = None,
) -> Link:
    """
    Apply decay to a single link.

    Args:
        link: The link to decay
        config: Decay configuration
        current_time: Current time (defaults to UTC now)

    Returns:
        The link with updated weight and status
    """
    current_time = current_time or datetime.now(timezone.utc)
    last_decay = link.weight.last_decay_at

    # Handle timezone-naive datetimes
    if last_decay.tzinfo is None:
        last_decay = last_decay.replace(tzinfo=timezone.utc)

    hours_elapsed = (current_time - last_decay).total_seconds() / 3600

    # Apply decay
    new_weight = apply_decay_to_weight(
        link.weight.value,
        config.decay_rate,
        hours_elapsed,
        config.minimum_weight,
    )

    link.weight.value = new_weight
    link.weight.last_decay_at = current_time

    # Update status based on new weight
    if new_weight <= config.minimum_weight:
        link.status = LinkStatus.ARCHIVED
    elif link.status == LinkStatus.VALIDATED and new_weight < link.weight.promotion_threshold:
        link.status = LinkStatus.DECAYING

    return link


def decay_links_batch(
    links: List[Link],
    config: LinkDecayConfig,
    current_time: Optional[datetime] = None,
) -> tuple[List[Link], int]:
    """
    Apply decay to a batch of links.

    Args:
        links: List of links to process
        config: Decay configuration
        current_time: Current time (defaults to UTC now)

    Returns:
        Tuple of (processed links, count of links that were decayed)
    """
    if not config.enabled:
        return links, 0

    current_time = current_time or datetime.now(timezone.utc)
    decayed_count = 0

    for link in links:
        if should_decay(link, config, current_time):
            decay_link(link, config, current_time)
            decayed_count += 1

    return links, decayed_count


def calculate_half_life_hours(decay_rate: float) -> float:
    """
    Calculate the half-life in hours for a given decay rate.

    Half-life is the time it takes for a link weight to decay to 50% of its value.

    Args:
        decay_rate: The decay rate per hour

    Returns:
        Half-life in hours
    """
    if decay_rate <= 0:
        return float("inf")
    return math.log(2) / decay_rate


def calculate_time_to_threshold(
    current_weight: float,
    target_weight: float,
    decay_rate: float,
) -> float:
    """
    Calculate hours until weight decays to a target threshold.

    Args:
        current_weight: Current weight value
        target_weight: Target weight threshold
        decay_rate: Decay rate per hour

    Returns:
        Hours until weight reaches target (inf if already below)
    """
    if current_weight <= target_weight:
        return 0.0
    if decay_rate <= 0:
        return float("inf")

    # From: target = current * exp(-rate * time)
    # Solve for time: time = -ln(target/current) / rate
    return -math.log(target_weight / current_weight) / decay_rate
