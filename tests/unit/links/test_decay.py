"""Tests for link decay functionality."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from akm.core.config import LinkDecayConfig
from akm.core.models import Link, LinkStatus, LinkWeight
from akm.links.decay import (
    apply_decay_to_weight,
    calculate_decay_factor,
    calculate_half_life_hours,
    calculate_time_to_threshold,
    decay_link,
    should_decay,
)


class TestDecayFactor:
    """Test decay factor calculations."""

    def test_zero_time_no_decay(self) -> None:
        """Zero time elapsed means no decay."""
        factor = calculate_decay_factor(0.1, 0.0)
        assert factor == 1.0

    def test_positive_decay(self) -> None:
        """Positive time and rate should reduce factor."""
        factor = calculate_decay_factor(0.1, 10.0)
        assert 0 < factor < 1.0

    def test_higher_rate_faster_decay(self) -> None:
        """Higher decay rate should result in smaller factor."""
        slow = calculate_decay_factor(0.01, 10.0)
        fast = calculate_decay_factor(0.1, 10.0)
        assert fast < slow

    def test_longer_time_more_decay(self) -> None:
        """Longer time should result in smaller factor."""
        short = calculate_decay_factor(0.1, 1.0)
        long = calculate_decay_factor(0.1, 10.0)
        assert long < short

    def test_zero_rate_no_decay(self) -> None:
        """Zero decay rate should preserve weight."""
        factor = calculate_decay_factor(0.0, 100.0)
        assert factor == 1.0


class TestApplyDecayToWeight:
    """Test applying decay to weight values."""

    def test_decay_reduces_weight(self) -> None:
        """Decay should reduce the weight."""
        original = 1.0
        new = apply_decay_to_weight(original, 0.1, 10.0)
        assert new < original

    def test_minimum_weight_enforced(self) -> None:
        """Weight should not go below minimum."""
        new = apply_decay_to_weight(0.1, 1.0, 1000.0, minimum_weight=0.05)
        assert new >= 0.05

    def test_never_negative(self) -> None:
        """Weight should never go negative."""
        new = apply_decay_to_weight(0.01, 10.0, 1000.0)
        assert new >= 0.0


class TestShouldDecay:
    """Test decay eligibility checks."""

    def test_decay_disabled(self) -> None:
        """No decay when disabled."""
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
        )
        config = LinkDecayConfig(enabled=False)
        assert should_decay(link, config) is False

    def test_archived_links_dont_decay(self) -> None:
        """Archived links should not decay further."""
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            status=LinkStatus.ARCHIVED,
        )
        config = LinkDecayConfig(enabled=True, decay_interval_hours=1)
        assert should_decay(link, config) is False

    def test_decay_interval_respected(self) -> None:
        """Decay should only happen after interval."""
        now = datetime.now(timezone.utc)
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            weight=LinkWeight(last_decay_at=now),
        )
        config = LinkDecayConfig(enabled=True, decay_interval_hours=24)

        # Just created - should not decay yet
        assert should_decay(link, config, now) is False

        # After interval - should decay
        future = now + timedelta(hours=25)
        assert should_decay(link, config, future) is True


class TestDecayLink:
    """Test link decay application."""

    def test_decay_reduces_weight(self) -> None:
        """Decay should reduce link weight."""
        past = datetime.now(timezone.utc) - timedelta(hours=100)
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            weight=LinkWeight(value=1.0, last_decay_at=past, decay_rate=0.01),
        )
        config = LinkDecayConfig(enabled=True, decay_rate=0.01, minimum_weight=0.1)

        decay_link(link, config)
        assert link.weight.value < 1.0

    def test_decay_archives_low_weight(self) -> None:
        """Links should be archived when weight drops too low."""
        past = datetime.now(timezone.utc) - timedelta(hours=1000)
        link = Link(
            source_id="00000000-0000-0000-0000-000000000001",
            target_id="00000000-0000-0000-0000-000000000002",
            weight=LinkWeight(value=0.15, last_decay_at=past, decay_rate=0.1),
        )
        config = LinkDecayConfig(enabled=True, decay_rate=0.1, minimum_weight=0.1)

        decay_link(link, config)
        assert link.status == LinkStatus.ARCHIVED


class TestHalfLife:
    """Test half-life calculations."""

    def test_half_life_calculation(self) -> None:
        """Half-life should be ln(2) / decay_rate."""
        decay_rate = 0.1
        half_life = calculate_half_life_hours(decay_rate)
        expected = math.log(2) / decay_rate
        assert abs(half_life - expected) < 0.001

    def test_zero_rate_infinite_half_life(self) -> None:
        """Zero decay rate means infinite half-life."""
        half_life = calculate_half_life_hours(0.0)
        assert half_life == float("inf")


class TestTimeToThreshold:
    """Test time-to-threshold calculations."""

    def test_already_below_threshold(self) -> None:
        """Already below threshold should return 0."""
        time = calculate_time_to_threshold(0.1, 0.2, 0.1)
        assert time == 0.0

    def test_zero_rate_infinite_time(self) -> None:
        """Zero decay rate means infinite time."""
        time = calculate_time_to_threshold(1.0, 0.5, 0.0)
        assert time == float("inf")

    def test_positive_time_calculation(self) -> None:
        """Should calculate correct time to reach threshold."""
        # With decay rate 0.1 per hour, to go from 1.0 to 0.5:
        # 0.5 = 1.0 * exp(-0.1 * t)
        # t = -ln(0.5) / 0.1 = ln(2) / 0.1
        time = calculate_time_to_threshold(1.0, 0.5, 0.1)
        expected = math.log(2) / 0.1
        assert abs(time - expected) < 0.001
