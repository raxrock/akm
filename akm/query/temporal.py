"""Decision lineage and temporal queries for the AKM framework."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from akm.core.interfaces import GraphBackend
from akm.core.models import Entity, Link, LinkStatus, Relationship

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """Represents an event in an entity or link timeline."""

    timestamp: datetime
    event_type: str  # "created", "updated", "validated", "decayed", "promoted", etc.
    entity_id: Optional[str] = None
    link_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "TimelineEvent") -> bool:
        return self.timestamp < other.timestamp


@dataclass
class LinkEvolution:
    """Tracks the evolution of a link over time."""

    link_id: str
    source_entity_name: str
    target_entity_name: str
    events: List[TimelineEvent] = field(default_factory=list)
    weight_history: List[Tuple[datetime, float]] = field(default_factory=list)

    @property
    def current_weight(self) -> float:
        """Get the most recent weight."""
        if self.weight_history:
            return self.weight_history[-1][1]
        return 0.0

    @property
    def weight_trend(self) -> str:
        """Determine if weight is trending up, down, or stable."""
        if len(self.weight_history) < 2:
            return "stable"

        recent = self.weight_history[-1][1]
        previous = self.weight_history[-2][1]

        if recent > previous + 0.05:
            return "increasing"
        elif recent < previous - 0.05:
            return "decreasing"
        return "stable"


@dataclass
class DecisionLineage:
    """Represents the lineage of a decision or knowledge evolution."""

    root_entity: Entity
    timeline: List[TimelineEvent] = field(default_factory=list)
    involved_entities: List[Entity] = field(default_factory=list)
    link_evolutions: List[LinkEvolution] = field(default_factory=list)
    key_milestones: List[TimelineEvent] = field(default_factory=list)

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[TimelineEvent]:
        """Get events within a time range."""
        return [e for e in self.timeline if start <= e.timestamp <= end]


@dataclass
class TemporalQueryResult:
    """Result from a temporal query."""

    query: str
    time_range: Optional[Tuple[datetime, datetime]] = None
    events: List[TimelineEvent] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    lineage: Optional[DecisionLineage] = None
    summary: str = ""


class TemporalQueryEngine:
    """
    Engine for temporal and decision lineage queries.

    This engine supports:
    1. Querying entity/link evolution over time
    2. Building decision lineage graphs
    3. Understanding how knowledge changed
    4. Finding patterns in temporal data
    """

    def __init__(
        self,
        graph: GraphBackend,
    ) -> None:
        """
        Initialize the temporal query engine.

        Args:
            graph: Graph backend for entity access
        """
        self._graph = graph
        self._link_cache: Dict[str, Link] = {}

    def set_link_cache(self, links: Dict[str, Link]) -> None:
        """Set the link cache for temporal analysis."""
        self._link_cache = links

    def get_entity_timeline(
        self,
        entity_id: str,
        links: Optional[List[Link]] = None,
    ) -> List[TimelineEvent]:
        """
        Get the timeline of events for an entity.

        Args:
            entity_id: Entity ID
            links: Optional list of related links

        Returns:
            List of timeline events sorted by time
        """
        events = []

        entity = self._graph.get_entity(entity_id)
        if not entity:
            return events

        # Entity creation event
        events.append(
            TimelineEvent(
                timestamp=entity.created_at,
                event_type="created",
                entity_id=entity_id,
                description=f"Entity '{entity.name}' was created",
                metadata={
                    "entity_type": entity.entity_type.value if hasattr(entity.entity_type, "value") else entity.entity_type,
                },
            )
        )

        # Entity update event (if different from creation)
        if entity.updated_at != entity.created_at:
            events.append(
                TimelineEvent(
                    timestamp=entity.updated_at,
                    event_type="updated",
                    entity_id=entity_id,
                    description=f"Entity '{entity.name}' was updated",
                )
            )

        # Link events
        if links:
            for link in links:
                if str(link.source_id) == entity_id or str(link.target_id) == entity_id:
                    events.extend(self._get_link_events(link, entity.name))

        events.sort()
        return events

    def get_link_evolution(
        self,
        link: Link,
        source_name: str = "Source",
        target_name: str = "Target",
    ) -> LinkEvolution:
        """
        Get the evolution of a link over time.

        Args:
            link: The link to analyze
            source_name: Name of source entity
            target_name: Name of target entity

        Returns:
            Link evolution data
        """
        evolution = LinkEvolution(
            link_id=str(link.id),
            source_entity_name=source_name,
            target_entity_name=target_name,
        )

        # Add creation event
        evolution.events.append(
            TimelineEvent(
                timestamp=link.created_at,
                event_type="link_created",
                link_id=str(link.id),
                description=f"Link between '{source_name}' and '{target_name}' was created",
                metadata={
                    "link_type": link.link_type,
                    "initial_weight": link.weight.initial_value,
                },
            )
        )
        evolution.weight_history.append((link.created_at, link.weight.initial_value))

        # Add validation events
        if link.weight.validation_count > 0:
            validation_time = link.weight.last_validated_at or link.updated_at
            evolution.events.append(
                TimelineEvent(
                    timestamp=validation_time,
                    event_type="validated",
                    link_id=str(link.id),
                    description=f"Link was validated ({link.weight.positive_validations} positive, {link.weight.negative_validations} negative)",
                    metadata={
                        "positive": link.weight.positive_validations,
                        "negative": link.weight.negative_validations,
                    },
                )
            )

        # Add decay event if applicable
        if link.weight.last_decay_at != link.created_at:
            evolution.events.append(
                TimelineEvent(
                    timestamp=link.weight.last_decay_at,
                    event_type="decayed",
                    link_id=str(link.id),
                    description=f"Link weight decayed to {link.weight.value:.3f}",
                )
            )
            evolution.weight_history.append((link.weight.last_decay_at, link.weight.value))

        # Add current weight
        evolution.weight_history.append((datetime.utcnow(), link.weight.value))

        # Add status change events
        if link.status == LinkStatus.VALIDATED:
            evolution.events.append(
                TimelineEvent(
                    timestamp=link.updated_at,
                    event_type="promoted",
                    link_id=str(link.id),
                    description="Link was promoted to validated status",
                )
            )
        elif link.status == LinkStatus.ARCHIVED:
            evolution.events.append(
                TimelineEvent(
                    timestamp=link.updated_at,
                    event_type="archived",
                    link_id=str(link.id),
                    description="Link was archived due to low weight",
                )
            )

        evolution.events.sort()
        return evolution

    def build_decision_lineage(
        self,
        entity_id: str,
        links: Optional[List[Link]] = None,
        max_depth: int = 3,
    ) -> DecisionLineage:
        """
        Build the decision lineage for an entity.

        Args:
            entity_id: Starting entity ID
            links: Optional list of links to include
            max_depth: Maximum traversal depth

        Returns:
            Decision lineage data
        """
        entity = self._graph.get_entity(entity_id)
        if not entity:
            return DecisionLineage(
                root_entity=Entity(name="Unknown", id=UUID(entity_id)),
            )

        lineage = DecisionLineage(root_entity=entity)

        # Get timeline for root entity
        lineage.timeline = self.get_entity_timeline(entity_id, links)

        # Traverse to find involved entities
        visited = {entity_id}
        to_visit = [(entity_id, 0)]

        while to_visit:
            current_id, depth = to_visit.pop(0)

            if depth >= max_depth:
                continue

            # Get neighbors through relationships
            neighbors = self._graph.get_neighbors(current_id, depth=1)
            for neighbor in neighbors:
                if str(neighbor.id) not in visited:
                    visited.add(str(neighbor.id))
                    lineage.involved_entities.append(neighbor)
                    to_visit.append((str(neighbor.id), depth + 1))

                    # Add neighbor events to timeline
                    lineage.timeline.append(
                        TimelineEvent(
                            timestamp=neighbor.created_at,
                            event_type="related_entity_created",
                            entity_id=str(neighbor.id),
                            description=f"Related entity '{neighbor.name}' was created",
                        )
                    )

        # Analyze links
        if links:
            entity_ids = visited
            for link in links:
                if str(link.source_id) in entity_ids or str(link.target_id) in entity_ids:
                    source_entity = self._graph.get_entity(str(link.source_id))
                    target_entity = self._graph.get_entity(str(link.target_id))

                    source_name = source_entity.name if source_entity else "Unknown"
                    target_name = target_entity.name if target_entity else "Unknown"

                    evolution = self.get_link_evolution(link, source_name, target_name)
                    lineage.link_evolutions.append(evolution)
                    lineage.timeline.extend(evolution.events)

        # Sort timeline and identify key milestones
        lineage.timeline.sort()
        lineage.key_milestones = self._identify_milestones(lineage.timeline)

        return lineage

    def query_temporal(
        self,
        question: str,
        entity_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        links: Optional[List[Link]] = None,
    ) -> TemporalQueryResult:
        """
        Execute a temporal query.

        Args:
            question: Natural language question
            entity_id: Optional entity to focus on
            time_range: Optional time range to filter
            links: Optional links to include

        Returns:
            Temporal query result
        """
        result = TemporalQueryResult(
            query=question,
            time_range=time_range,
        )

        if entity_id:
            # Build lineage for specific entity
            lineage = self.build_decision_lineage(entity_id, links)
            result.lineage = lineage
            result.entities = [lineage.root_entity] + lineage.involved_entities
            result.events = lineage.timeline

            # Filter by time range if specified
            if time_range:
                start, end = time_range
                result.events = [e for e in result.events if start <= e.timestamp <= end]
                result.entities = [
                    e for e in result.entities
                    if start <= e.created_at <= end
                ]
        else:
            # General temporal query - get recent events
            if time_range:
                result.events = self._get_events_in_range(time_range[0], time_range[1], links)
            else:
                # Default to last 30 days
                end = datetime.utcnow()
                start = end - timedelta(days=30)
                result.events = self._get_events_in_range(start, end, links)

        # Include relevant links
        if links:
            if time_range:
                start, end = time_range
                result.links = [
                    l for l in links
                    if start <= l.created_at <= end
                ]
            else:
                result.links = links

        # Generate summary
        result.summary = self._generate_temporal_summary(result)

        return result

    def find_changes_since(
        self,
        since: datetime,
        entity_ids: Optional[List[str]] = None,
        links: Optional[List[Link]] = None,
    ) -> Dict[str, Any]:
        """
        Find all changes since a given time.

        Args:
            since: Starting timestamp
            entity_ids: Optional list of entity IDs to check
            links: Optional list of links to check

        Returns:
            Dictionary of changes
        """
        changes = {
            "entities_created": [],
            "entities_updated": [],
            "links_created": [],
            "links_validated": [],
            "links_decayed": [],
        }

        # Check entities
        if entity_ids:
            for entity_id in entity_ids:
                entity = self._graph.get_entity(entity_id)
                if entity:
                    if entity.created_at >= since:
                        changes["entities_created"].append({
                            "id": str(entity.id),
                            "name": entity.name,
                            "created_at": entity.created_at.isoformat(),
                        })
                    elif entity.updated_at >= since:
                        changes["entities_updated"].append({
                            "id": str(entity.id),
                            "name": entity.name,
                            "updated_at": entity.updated_at.isoformat(),
                        })

        # Check links
        if links:
            for link in links:
                if link.created_at >= since:
                    changes["links_created"].append({
                        "id": str(link.id),
                        "type": link.link_type,
                        "weight": link.weight.value,
                    })

                if link.weight.last_validated_at and link.weight.last_validated_at >= since:
                    changes["links_validated"].append({
                        "id": str(link.id),
                        "validations": link.weight.validation_count,
                    })

                if link.weight.last_decay_at >= since and link.weight.last_decay_at != link.created_at:
                    changes["links_decayed"].append({
                        "id": str(link.id),
                        "weight": link.weight.value,
                    })

        return changes

    def get_link_strength_over_time(
        self,
        link: Link,
        intervals: int = 10,
    ) -> List[Tuple[datetime, float]]:
        """
        Estimate link weight over time intervals.

        Args:
            link: The link to analyze
            intervals: Number of time intervals

        Returns:
            List of (timestamp, weight) tuples
        """
        history = []
        start_time = link.created_at
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        interval_seconds = duration / intervals

        # Simple linear interpolation from initial to current weight
        for i in range(intervals + 1):
            t = start_time + timedelta(seconds=interval_seconds * i)
            progress = i / intervals

            # Approximate weight at this time
            # This is simplified - real implementation would track actual changes
            if link.weight.validation_count > 0:
                # Weight increased due to validations
                weight = link.weight.initial_value + (link.weight.value - link.weight.initial_value) * progress
            else:
                # Weight decayed over time
                weight = link.weight.initial_value - (link.weight.initial_value - link.weight.value) * progress

            history.append((t, max(0, min(1, weight))))

        return history

    def _get_link_events(self, link: Link, entity_name: str) -> List[TimelineEvent]:
        """Get timeline events for a link."""
        events = []

        # Link creation
        events.append(
            TimelineEvent(
                timestamp=link.created_at,
                event_type="link_created",
                link_id=str(link.id),
                description=f"Link created involving '{entity_name}'",
                metadata={"link_type": link.link_type},
            )
        )

        # Validations
        if link.weight.last_validated_at:
            events.append(
                TimelineEvent(
                    timestamp=link.weight.last_validated_at,
                    event_type="link_validated",
                    link_id=str(link.id),
                    description=f"Link validated ({link.weight.validation_count} times)",
                )
            )

        return events

    def _get_events_in_range(
        self,
        start: datetime,
        end: datetime,
        links: Optional[List[Link]] = None,
    ) -> List[TimelineEvent]:
        """Get all events in a time range."""
        events = []

        if links:
            for link in links:
                if start <= link.created_at <= end:
                    events.append(
                        TimelineEvent(
                            timestamp=link.created_at,
                            event_type="link_created",
                            link_id=str(link.id),
                            description=f"Link of type '{link.link_type}' was created",
                        )
                    )

        events.sort()
        return events

    def _identify_milestones(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Identify key milestones in a timeline."""
        milestones = []

        milestone_types = {"created", "promoted", "validated", "archived"}

        for event in events:
            if event.event_type in milestone_types:
                milestones.append(event)

        return milestones

    def _generate_temporal_summary(self, result: TemporalQueryResult) -> str:
        """Generate a summary of temporal query results."""
        parts = [f"Temporal analysis for: '{result.query}'", ""]

        if result.time_range:
            start, end = result.time_range
            parts.append(f"Time range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
            parts.append("")

        if result.events:
            parts.append(f"Found {len(result.events)} events:")
            event_types = {}
            for event in result.events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

            for event_type, count in sorted(event_types.items()):
                parts.append(f"  - {event_type}: {count}")
            parts.append("")

        if result.lineage and result.lineage.link_evolutions:
            parts.append(f"Link evolutions: {len(result.lineage.link_evolutions)}")
            for evolution in result.lineage.link_evolutions[:5]:
                parts.append(
                    f"  - {evolution.source_entity_name} <-> {evolution.target_entity_name}: "
                    f"weight {evolution.current_weight:.2f} ({evolution.weight_trend})"
                )

        return "\n".join(parts)
