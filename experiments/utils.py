import argparse
from typing import Dict, Any


def add_arguments(
    parser: argparse.ArgumentParser, arguments: Dict[str, Dict[str, Any]]
):
    """Adds arguments from `arguments` to `parser`.

    Args:
        parser: CL argument parser.
        arguments: a dictionary with name of variable
            as key and kwargs as value.
    """
    for k, v in arguments.items():
        parser.add_argument(f"--{k}", **v)


general_argparse_args = dict(
    reps=dict(default=1, type=int, help="times to run experiment"),
    description=dict(
        type=str,
        help="additional information on the experiment for the handler, "
        "e.g. internal code change not reflected in the hyperparams",
    ),
    logging_level=dict(
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "CRITICAL", "ERROR", "FATAL"],
        help="level of logging module",
    ),
    logging_file=dict(
        type=str,
        help="where to log results, default is stderr",
    ),
)
