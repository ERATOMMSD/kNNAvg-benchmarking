from pathlib import Path

import pandas as pd
import nmoo
from pymoo.core.problem import Problem
from pymoo.factory import get_problem, get_reference_directions
from numpy import array

from pymoo.problems.multi import MODAct
from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated
from pymoo.problems.multi.omnitest import OmniTest


def get_pareto_front(problem: nmoo.WrappedProblem) -> array:
    name = problem.ground_problem().name().lower()
    if name.lower() == "mw4":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        pareto_front = problem.ground_problem().pareto_front(ref_dirs)
    elif name.lower() == "mw8":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=15)
        pareto_front = problem.ground_problem().pareto_front(ref_dirs)
    else:
        pareto_front = problem.ground_problem().pareto_front()
    return pareto_front


def get_hv_refpoint(problem: nmoo.WrappedProblem) -> array:
    name = problem.ground_problem().name().lower()
    ref_point_file = Path("hv_refpoints.csv")
    assert ref_point_file.exists()
    ref_point_df = pd.read_csv(ref_point_file)

    if name == "omnitest":
        name = f"omnitest_{problem.n_var}"

    refpoint_filter = ref_point_df[ref_point_df["problem"] == name]["ref_point"]
    assert len(refpoint_filter) == 1, f"Asserting that there is exactly one reference point for {name}, but we found {len(refpoint_filter)}"

    return eval(refpoint_filter.iloc[0])


def label_to_problem(label: str) -> Problem:
    """ Magic function that turns a string to a problem as we want it! """
    label = label.lower()

    if label.startswith("dtlz") or label.startswith("wfg"):
        splits = label.split("_")
        if len(splits) == 1:
            return get_problem(splits[0])
        elif len(splits) == 2:
            return get_problem(splits[0], n_var=int(splits[1]))
        else:
            return get_problem(splits[0], n_var=int(splits[1]), n_obj=int(splits[2]))

    # the part before the underscore is the problem name, the part after (if exists) the n_var
    if label.startswith("zdt"):
        splits = label.split("_")
        if len(splits) == 1:
            return get_problem(splits[0])
        else:
            return get_problem(splits[0], n_var=int(splits[1]))

    if label.startswith("mw") or label.startswith("ctp"):
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

    if label.startswith("cs"):
        splits = label.split("_")
        if len(splits) == 1:
            return MODAct(splits[0])
        else:
            return MODAct(splits[0], )

    if label in ["bnh", "kursawe", "osy", "tnk", "truss2d", "welded_beam"]:
        return get_problem(label)

    if label == "sympart":
        return SYMPART()

    if label == "sympartrotated":
        return SYMPARTRotated()
