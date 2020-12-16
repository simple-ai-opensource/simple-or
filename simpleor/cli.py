"""Console script for simpleor."""
import sys
import click
import logging

from simpleor.utils import PROJECT_DIRECTORY
from simpleor.monitoring import _configure_logger, LOGGING_LEVELS
from simpleor.scheduler import (
    READ_OPTIONS,
    WRITE_OPTIONS,
    read_schedule_problem,
    ScheduleSolver,
)

logger = logging.getLogger(__name__)


# General
verbose_option = click.option(
    "-v", "--verbose", default="info", help=f"Verbosity level {LOGGING_LEVELS}"
)

# Scheduler cli
task_durations_path_option = click.option(
    "--durationsfile",
    type=str,
    default=str(PROJECT_DIRECTORY.joinpath("data/scheduler/task_durations.csv")),
    help="Path to task_durations file",
)

available_schedule_path_option = click.option(
    "--schedulefile",
    type=str,
    default=str(PROJECT_DIRECTORY.joinpath("data/scheduler/available_schedule.csv")),
    help="Path to available_schedule file",
)

read_kind_option = click.option(
    "--read",
    type=str,
    default="csv",
    help=f"What kind of file to read ({READ_OPTIONS})",
)

solution_directory_option = click.option(
    "--solutiondir",
    type=str,
    default=str(PROJECT_DIRECTORY.joinpath("data/scheduler")),
    help="Directory where the solution is written to",
)

solution_filename_option = click.option(
    "--solutionfile",
    type=str,
    default="solution_cli.csv",
    help="Filename of the solution",
)

solution_write_kind_option = click.option(
    "--write",
    type=str,
    default="csv",
    help=f"What kind of file to read ({WRITE_OPTIONS})",
)


@click.command()
@task_durations_path_option
@available_schedule_path_option
@read_kind_option
@solution_directory_option
@solution_filename_option
@solution_write_kind_option
@verbose_option
def schedule(
    durationsfile: str,
    schedulefile: str,
    read: str,
    solutiondir: str,
    solutionfile: str,
    write: str,
    verbose: str,
):
    """Console script for clients."""
    _configure_logger(verbose=verbose)
    logger.info("Running scheduler...")
    task_durations, available_timeslots = read_schedule_problem(
        task_durations_file_path=durationsfile,
        available_schedule_file_path=schedulefile,
        how=read,
    )
    solver = ScheduleSolver(
        task_durations=task_durations, available_timeslots=available_timeslots
    )
    solver.solve()
    solver.write_solution(directory=solutiondir, filename=solutionfile, how=write)
    logger.info("Scheduler successful!")
    return 0


if __name__ == "__main__":
    sys.exit(schedule())
