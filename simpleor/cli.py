"""Console script for simpleor."""
import sys
import click
import logging

from simpleor.base import _configure_logger

logger = logging.getLogger(__name__)


@click.command()
def main(args=None):
    """Console script for clients."""
    click.echo("Replace this message by putting your code into " "simpleor.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
