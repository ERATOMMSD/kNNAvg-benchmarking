import itertools
import os
import click

from pymoo.model.problem import Problem
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.factory import get_problem

# Deactivate pymoo warning!
from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

import logging

from noisy_system_pymoo import run_problem

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------

def label_to_problem(label: str) -> Problem:
    """ Magic function that turns a string to a problem as we want it! """
    label = label.lower()

    # the part before the underscore is the problem name, the part after (if exists) the n_var
    if label.startswith("zdt") or label.startswith("mw") or label.startswith("ctp"):
        splits = label.split("_")
        if len(splits) == 1:
            return get_problem(splits[0])
        else:
            return get_problem(splits[0], n_var=int(splits[1]))

    if label.startswith("omnitest") :
        splits = label.split("_")
        if len(splits) == 1:
            return OmniTest()
        else:
            return OmniTest(n_var=int(splits[1]))

    if label.startswith("dascmop"):
        splits = label.split("_")
        if len(splits) == 1:
            return get_problem(splits[0])
        else:
            return get_problem(splits[0], int(splits[1]))

    if label in ["bnh", "carside", "kursawe", "osy", "tnk", "truss2d", "welded beam", "sympart"]:
        return get_problem(label)

    logger.error(f"Cannot handle problem {label}")


# ---------------------------------------------------

@click.group()
def cli():
    pass

@cli.command('bench')
@click.argument('problem', type=str)
@click.option('--noise', default=0, type=float, help="The standard deviation for each dimension")
@click.option('--evals', default=-1, type=int)
def run_single(problem, noise, evals):
    return run_problem(problem, label_to_problem(problem), noise, evals)

@cli.command('all')
@click.option('--noise', default=0, type=float, help="The standard deviation for each dimension")
@click.option('--evals', default=-1, type=int)
def run_all(noise, evals):
    selection = [
        *[f"ZDT{problem_idx + 1}" for problem_idx in range(6)],
        *[f"MW{problem_idx + 1}" for problem_idx in range(14)],
        *[f"CTP{problem_idx + 1}" for problem_idx in range(8)],
        *[f"DASCMOP{problem_idx + 1}_{difficulty + 1}" for problem_idx, difficulty in itertools.product(range(12), range(16))],
        *["bnh", "carside", "kursawe", "osy", "tnk", "truss2d", "welded beam", "sympart"],
        *[f"OmniTest_{problem_idx + 1}" for problem_idx in range(1, 5)],  # NO PARETO FRONT !!!!
    ]

    print(f"Going to run {len(selection)} benchmarks")
    for problem_name in selection:
        try:
            print(f"Trying {problem_name} next")
            run_single.callback(problem_name, noise=noise, evals=evals)
        except:
            logger.warning(f"failed {problem_name}", exc_info=True)

if __name__ == "__main__":
    cli()