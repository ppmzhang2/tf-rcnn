"""The Region Proposal Network (RPN) CLI."""
import click

from rcnn import trainer


@click.command()
@click.option("--epochs",
              type=click.INT,
              default=5,
              help="number of epochs to train.")
@click.option("--save-intv",
              type=click.INT,
              default=10,
              help="number of batches between each save.")
@click.option("--batch", type=click.INT, default=4, help="batch size.")
def train_rpn(epochs: int, save_intv: int, batch: int) -> None:
    """Train the RPN."""
    return trainer.train_rpn(epochs, save_intv, batch)


@click.command()
@click.option("--images",
              type=click.INT,
              default=10,
              help="number of images to add bounding boxes to.")
def predict_rpn(images: int) -> None:
    """Predict with the RPN."""
    return trainer.predict_rpn(images)
