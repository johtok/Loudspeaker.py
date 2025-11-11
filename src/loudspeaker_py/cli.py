"""Command-line interface for Loudspeaker Py CFES framework."""

import click
from . import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Loudspeaker Py - CFES Testing Framework for Loudspeaker Modeling"""
    pass


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def train(config, verbose):
    """Train models using configuration file."""
    click.echo(f"Starting training with config: {config}")
    if verbose:
        click.echo("Verbose mode enabled")


@main.command()
@click.option('--model-path', type=click.Path(exists=True), help='Path to trained model')
@click.option('--data-path', type=click.Path(exists=True), help='Path to evaluation data')
@click.option('--output', '-o', type=click.Path(), help='Output path for results')
def evaluate(model_path, data_path, output):
    """Evaluate trained models."""
    click.echo(f"Evaluating model: {model_path}")
    click.echo(f"Using data: {data_path}")
    if output:
        click.echo(f"Results will be saved to: {output}")


@main.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Input data or model path')
@click.option('--output', '-o', type=click.Path(), help='Output path for visualization')
@click.option('--format', '-f', type=click.Choice(['png', 'pdf', 'html']), default='png', help='Output format')
def visualize(input, output, format):
    """Create visualizations."""
    click.echo(f"Creating visualization from: {input}")
    click.echo(f"Output format: {format}")
    if output:
        click.echo(f"Visualization will be saved to: {output}")


@main.command()
@click.option('--dataset', type=click.Choice(['spirals', 'lorenz', 'chua', 'custom']), required=True, help='Dataset to generate')
@click.option('--size', type=int, default=1000, help='Number of samples to generate')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output path for generated data')
def generate_data(dataset, size, output):
    """Generate synthetic datasets for testing."""
    click.echo(f"Generating {size} samples for dataset: {dataset}")
    click.echo(f"Data will be saved to: {output}")


if __name__ == '__main__':
    main()