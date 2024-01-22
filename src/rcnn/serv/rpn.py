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
@click.option("--images",
              type=click.INT,
              default=10,
              help="number of images to add bounding boxes to.")
def predict_rpn(images: int) -> None:
    """Predict with the RPN."""
    return trainer.predict_rpn(images)


@click.command()
@click.option("--n", type=click.INT, default=10, help="#images to show")
def show_gt(n: int) -> None:
    """Show GT."""
    return vis.show_gt(n)
