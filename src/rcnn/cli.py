"""All CLI commands are defined here."""
import click

from rcnn.serv import rpn as rpn_cli


@click.group()
def cli() -> None:
    """CLI for the Faster R-CNN."""


cli.add_command(rpn_cli.train_rpn)
cli.add_command(rpn_cli.predict_rpn)
cli.add_command(rpn_cli.show_gt)
cli.add_command(rpn_cli.show_tr)
