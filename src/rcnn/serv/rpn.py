"""The Region Proposal Network (RPN) CLI."""
import click

from rcnn import trainer
from rcnn import vis


@click.command()
@click.option("--epochs", type=click.INT, default=5, help="epochs to train.")
@click.option("--batch-size", type=click.INT, default=1, help="batch size.")
def train_rpn(epochs: int, batch_size: int) -> None:
    """Train the RPN."""
    return trainer.train_rpn(epochs, batch_size)


@click.command()
@click.option("--n", type=click.INT, default=10, help="#images to show")
def predict_rpn(n: int) -> None:
    """Predict with the RPN."""
    return trainer.predict_rpn(n)


@click.command()
@click.option("--n", type=click.INT, default=10, help="#images to show")
def show_gt(n: int) -> None:
    """Show GT."""
    return vis.show_gt(n)


@click.command()
@click.option("--n", type=click.INT, default=10, help="#images to show")
def show_tr(n: int) -> None:
    """Show Training Data."""
    return vis.show_tr(n)
