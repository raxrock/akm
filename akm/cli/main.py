"""AKM Command Line Interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from akm import __version__
from akm.api.client import AKM
from akm.core.config import AKMConfig

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="akm")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str]) -> None:
    """
    Adaptive Knowledge Mesh (AKM) CLI.

    A framework for building adaptive knowledge graphs with semantic search
    and intelligent link management.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option(
    "--name",
    "-n",
    default="akm_project",
    help="Project name",
)
@click.option(
    "--path",
    "-p",
    type=click.Path(),
    default=".",
    help="Project path",
)
@click.option(
    "--domain",
    "-d",
    type=click.Choice(["generic", "software_engineering", "research"]),
    default="generic",
    help="Domain adapter",
)
def init(name: str, path: str, domain: str) -> None:
    """Initialize a new AKM project."""
    project_path = Path(path)
    if project_path.name != name:
        project_path = project_path / name

    # Create project structure
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / ".akm").mkdir(exist_ok=True)
    (project_path / "data").mkdir(exist_ok=True)

    # Create default config
    config = AKMConfig(
        project_name=name,
        data_dir=str(project_path / ".akm"),
    )
    config.domain.name = domain
    config_file = project_path / "akm.yaml"
    config.to_yaml(config_file)

    console.print(f"[green]Project '{name}' initialized at {project_path}[/green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  cd {project_path}")
    console.print("  akm entity add 'MyEntity' --type concept")
    console.print("  akm link create <source_id> <target_id>")
    console.print("  akm stats")


@cli.group()
def entity() -> None:
    """Entity management commands."""
    pass


@entity.command("add")
@click.argument("name")
@click.option("--type", "-t", "entity_type", default="generic", help="Entity type")
@click.option("--description", "-d", help="Entity description")
@click.option("--properties", "-p", help="JSON properties")
@click.pass_context
def entity_add(
    ctx: click.Context,
    name: str,
    entity_type: str,
    description: Optional[str],
    properties: Optional[str],
) -> None:
    """Add a new entity."""
    config_path = ctx.obj.get("config_path")

    props = {}
    if properties:
        try:
            props = json.loads(properties)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON for properties[/red]")
            sys.exit(1)

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        entity = akm.add_entity(
            name=name,
            entity_type=entity_type,
            description=description,
            properties=props,
        )
        console.print(f"[green]Created entity:[/green] {entity.id}")
        console.print(f"  Name: {entity.name}")
        console.print(f"  Type: {entity.entity_type}")


@entity.command("list")
@click.option("--type", "-t", "entity_type", help="Filter by type")
@click.option("--limit", "-l", default=20, help="Limit results")
@click.pass_context
def entity_list(
    ctx: click.Context,
    entity_type: Optional[str],
    limit: int,
) -> None:
    """List entities."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        entities = akm.find_entities(entity_type=entity_type, limit=limit)

        if not entities:
            console.print("[yellow]No entities found[/yellow]")
            return

        table = Table(title="Entities")
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Description")

        for e in entities:
            table.add_row(
                str(e.id)[:8] + "...",
                e.name,
                str(e.entity_type),
                (e.description or "")[:50],
            )

        console.print(table)


@entity.command("get")
@click.argument("entity_id")
@click.pass_context
def entity_get(ctx: click.Context, entity_id: str) -> None:
    """Get entity details."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        entity = akm.get_entity(entity_id)
        if not entity:
            console.print(f"[red]Entity not found: {entity_id}[/red]")
            sys.exit(1)

        console.print(f"[bold]Entity: {entity.name}[/bold]")
        console.print(f"  ID: {entity.id}")
        console.print(f"  Type: {entity.entity_type}")
        console.print(f"  Description: {entity.description or 'N/A'}")
        console.print(f"  Confidence: {entity.confidence}")
        console.print(f"  Created: {entity.created_at}")
        if entity.properties:
            console.print(f"  Properties: {json.dumps(entity.properties, indent=2)}")


@cli.group()
def link() -> None:
    """Adaptive link management commands."""
    pass


@link.command("create")
@click.argument("source_id")
@click.argument("target_id")
@click.option("--type", "-t", "link_type", default="inferred", help="Link type")
@click.option("--confidence", "-c", default=0.5, help="Pattern confidence")
@click.option("--source", "-s", help="Pattern source")
@click.pass_context
def link_create(
    ctx: click.Context,
    source_id: str,
    target_id: str,
    link_type: str,
    confidence: float,
    source: Optional[str],
) -> None:
    """Create a soft link between entities."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        link = akm.create_soft_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            pattern_source=source,
            pattern_confidence=confidence,
        )
        console.print(f"[green]Created link:[/green] {link.id}")
        console.print(f"  Status: {link.status.value}")
        console.print(f"  Weight: {link.weight.value:.3f}")


@link.command("validate")
@click.argument("link_id")
@click.option("--positive/--negative", default=True, help="Validation type")
@click.option("--strength", "-s", type=float, help="Validation strength")
@click.pass_context
def link_validate(
    ctx: click.Context,
    link_id: str,
    positive: bool,
    strength: Optional[float],
) -> None:
    """Validate a link (strengthen or weaken)."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        link = akm.validate_link(link_id, is_positive=positive, strength=strength)
        action = "strengthened" if positive else "weakened"
        console.print(f"[green]Link {action}:[/green]")
        console.print(f"  New weight: {link.weight.value:.3f}")
        console.print(f"  Status: {link.status.value}")


@link.command("list")
@click.argument("entity_id")
@click.option("--min-weight", "-w", default=0.0, help="Minimum weight")
@click.pass_context
def link_list(
    ctx: click.Context,
    entity_id: str,
    min_weight: float,
) -> None:
    """List links for an entity."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        links = akm.get_links(entity_id, min_weight=min_weight)

        if not links:
            console.print("[yellow]No links found[/yellow]")
            return

        table = Table(title=f"Links for {entity_id[:8]}...")
        table.add_column("ID", style="dim")
        table.add_column("Target")
        table.add_column("Type")
        table.add_column("Weight")
        table.add_column("Status")

        for link in links:
            table.add_row(
                str(link.id)[:8] + "...",
                str(link.target_id)[:8] + "...",
                link.link_type,
                f"{link.weight.value:.3f}",
                link.status.value,
            )

        console.print(table)


@cli.command()
@click.pass_context
def decay(ctx: click.Context) -> None:
    """Run link weight decay."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        count = akm.run_link_decay()
        console.print(f"[green]Decay complete:[/green] {count} links processed")


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show project statistics."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        link_stats = akm.get_link_stats()

        console.print("[bold]AKM Project Statistics[/bold]")
        console.print()

        # Link stats
        console.print("[cyan]Links:[/cyan]")
        console.print(f"  Total: {link_stats['total_links']}")
        console.print(f"  Average weight: {link_stats['average_weight']:.3f}")
        console.print(f"  Entities with links: {link_stats['entities_with_links']}")

        if link_stats["status_distribution"]:
            console.print("  Status distribution:")
            for status, count in link_stats["status_distribution"].items():
                console.print(f"    {status}: {count}")


@cli.command()
@click.argument("entity_id")
@click.option("--depth", "-d", default=2, help="Traversal depth")
@click.option("--include-links/--no-links", default=True, help="Include adaptive links")
@click.pass_context
def traverse(
    ctx: click.Context,
    entity_id: str,
    depth: int,
    include_links: bool,
) -> None:
    """Traverse the graph from an entity."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        result = akm.traverse(entity_id, depth=depth, include_links=include_links)

        console.print(f"[bold]Traversal from {result.start_entity.name}[/bold]")
        console.print(f"  Depth: {result.depth_reached}")
        console.print(f"  Entities found: {len(result.entities)}")
        console.print(f"  Relationships: {len(result.relationships)}")
        console.print(f"  Links: {len(result.links)}")

        if result.entities:
            console.print()
            console.print("[cyan]Connected entities:[/cyan]")
            for e in result.entities[:10]:
                console.print(f"  - {e.name} ({e.entity_type})")
            if len(result.entities) > 10:
                console.print(f"  ... and {len(result.entities) - 10} more")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", default=True, help="Scan recursively")
@click.option("--patterns", "-p", multiple=True, help="File patterns to include")
@click.option("--no-index", is_flag=True, help="Skip vector indexing")
@click.option("--no-links", is_flag=True, help="Skip soft link creation")
@click.pass_context
def ingest(
    ctx: click.Context,
    path: str,
    recursive: bool,
    patterns: tuple,
    no_index: bool,
    no_links: bool,
) -> None:
    """Ingest documents into the knowledge graph."""
    from akm.ingestion import IngestionPipeline
    from akm.ingestion.pipeline import IngestionConfig

    config_path = ctx.obj.get("config_path")

    config = IngestionConfig(
        recursive=recursive,
        patterns=list(patterns) if patterns else None,
        index_documents=not no_index,
        create_soft_links=not no_links,
    )

    def on_progress(processed: int, total: int) -> None:
        console.print(f"  Processed {processed} files...", end="\r")

    config.on_progress = on_progress

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        pipeline = IngestionPipeline(
            graph=akm.graph,
            vector=akm._vector,
            link_manager=akm._link_manager,
            domain_transformer=akm._domain_transformer,
            llm_provider=akm._llm,
            embedding_model=akm._embedding,
            collection_name=akm._config.vector.chromadb.collection_name if akm._config.vector.backend == "chromadb" else "akm_documents",
        )

        console.print(f"[cyan]Ingesting from {path}...[/cyan]")
        stats = pipeline.ingest(path, config)

        console.print()
        console.print("[green]Ingestion complete![/green]")
        console.print(f"  Files processed: {stats.files_processed}")
        console.print(f"  Files skipped: {stats.files_skipped}")
        console.print(f"  Files failed: {stats.files_failed}")
        console.print(f"  Entities created: {stats.entities_created}")
        console.print(f"  Relationships created: {stats.relationships_created}")
        console.print(f"  Soft links created: {stats.soft_links_created}")
        console.print(f"  Documents indexed: {stats.documents_indexed}")

        if stats.errors:
            console.print()
            console.print("[yellow]Errors:[/yellow]")
            for error in stats.errors[:5]:
                console.print(f"  - {error}")
            if len(stats.errors) > 5:
                console.print(f"  ... and {len(stats.errors) - 5} more")


@cli.command()
@click.argument("question")
@click.option("--max-hops", "-h", default=2, help="Maximum graph traversal hops")
@click.pass_context
def query(ctx: click.Context, question: str, max_hops: int) -> None:
    """Query the knowledge graph using natural language."""
    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        console.print(f"[cyan]Querying: {question}[/cyan]")
        console.print()

        result = akm.query(question, max_hops=max_hops)

        console.print("[bold]Answer:[/bold]")
        console.print(result.answer)
        console.print()
        console.print(f"Confidence: {result.confidence:.2f}")

        if result.entities_involved:
            console.print()
            console.print("[cyan]Entities involved:[/cyan]")
            for entity in result.entities_involved[:5]:
                console.print(f"  - {entity.name}")

        if result.reasoning_path:
            console.print()
            console.print("[cyan]Reasoning path:[/cyan]")
            for step in result.reasoning_path:
                console.print(f"  {step}")


@cli.group()
def train() -> None:
    """Model training commands."""
    pass


@train.command("link-prediction")
@click.option("--epochs", "-e", default=100, help="Number of training epochs")
@click.option("--learning-rate", "-lr", default=0.01, help="Learning rate")
@click.option("--hidden-channels", "-hc", default=64, help="Hidden layer size")
@click.pass_context
def train_link_prediction(
    ctx: click.Context,
    epochs: int,
    learning_rate: float,
    hidden_channels: int,
) -> None:
    """Train link prediction model."""
    from akm.gnn import GNNManager

    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        console.print("[cyan]Training link prediction model...[/cyan]")

        gnn = GNNManager(graph=akm.graph, config=akm.config.gnn)

        result = gnn.train_link_prediction(
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_channels=hidden_channels,
        )

        console.print()
        console.print("[green]Training complete![/green]")
        console.print(f"  Model type: {result.model_type}")
        console.print(f"  Epochs: {result.epochs}")
        console.print(f"  Final loss: {result.final_loss:.4f}")
        console.print(f"  Training time: {result.training_time_seconds:.2f}s")


@train.command("predict")
@click.option("--top-k", "-k", default=10, help="Number of predictions")
@click.option("--min-prob", "-p", default=0.5, help="Minimum probability")
@click.pass_context
def train_predict(ctx: click.Context, top_k: int, min_prob: float) -> None:
    """Predict new links using trained model."""
    from akm.gnn import GNNManager

    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        gnn = GNNManager(graph=akm.graph, config=akm.config.gnn)

        # Train first (or load from cache in future)
        console.print("[cyan]Training model for predictions...[/cyan]")
        gnn.train_link_prediction(epochs=50)

        console.print("[cyan]Predicting new links...[/cyan]")
        predictions = gnn.predict_links(top_k=top_k, min_probability=min_prob)

        if not predictions:
            console.print("[yellow]No link predictions found[/yellow]")
            return

        table = Table(title="Predicted Links")
        table.add_column("Source")
        table.add_column("Target")
        table.add_column("Probability")

        for pred in predictions:
            table.add_row(
                pred.source_name or pred.source_id[:8],
                pred.target_name or pred.target_id[:8],
                f"{pred.probability:.3f}",
            )

        console.print(table)


@cli.command()
@click.option("--resolution", "-r", default=1.0, help="Community resolution")
@click.option("--min-size", "-s", default=2, help="Minimum community size")
@click.pass_context
def communities(ctx: click.Context, resolution: float, min_size: int) -> None:
    """Detect communities in the knowledge graph."""
    from akm.gnn import GNNManager

    config_path = ctx.obj.get("config_path")

    with AKM.from_config(config_path) if config_path else AKM() as akm:
        gnn = GNNManager(graph=akm.graph, config=akm.config.gnn)

        console.print("[cyan]Detecting communities...[/cyan]")
        detected = gnn.detect_communities(
            resolution=resolution,
            min_community_size=min_size,
        )

        if not detected:
            console.print("[yellow]No communities detected[/yellow]")
            return

        console.print(f"[green]Found {len(detected)} communities[/green]")
        console.print()

        for i, community in enumerate(detected[:10]):
            console.print(f"[bold]Community {i + 1}: {community.label}[/bold]")
            console.print(f"  Size: {len(community.entity_ids)} entities")
            console.print(f"  Members: {', '.join(community.entity_names[:5])}")
            if len(community.entity_names) > 5:
                console.print(f"    ... and {len(community.entity_names) - 5} more")
            console.print()


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
