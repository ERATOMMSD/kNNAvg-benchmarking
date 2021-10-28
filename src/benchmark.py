# Cédric's benchmark following the TestGen meeting of Oct. 6th
# https://docs.google.com/document/d/1u_aW9MJXIkrTs3DJbXoQ-dvrWZc0CW69YYCDh4W8_Z0/edit#heading=h.kto0noy3m605

# To run on a big boy

import logging
import os
from itertools import product
from pathlib import Path

import click
import numpy as np
import pandas as pd
from nmoo.benchmark import Benchmark
from nmoo.denoisers import KNNAvg, ResampleAverage
from nmoo.evaluators import EvaluationPenaltyEvaluator
from nmoo.noises import GaussianNoise
from nmoo.plotting import plot_performance_indicators
from nmoo.wrapped_problem import WrappedProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi.zdt import ZDT1
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

from c_region_simulator_problem import CRegionSimulatorProblem
from pendulum_cart_problem import PendulumCartProblem

C_REGION_SIMULATOR_PATH = (
    Path(__file__).absolute().parent
    / "c_region_simulator_problem"
    / "c_region_simulator_with_pipe"
)
OUTPUT_DIR_PATH = (
    Path(__file__).absolute().parent / ".." / "results" / "benchmark2"
)
PLOTS_OUTPUT_DIR_PATH = OUTPUT_DIR_PATH / "plots"


@click.group()
def main() -> None:
    pass


@main.command()
def consolidate() -> None:
    """
    Performs post-processing on all available raw data.

    Computes global Pareto populations, performance indicators, and
    consolidates everything into benchmark.csv. Can be safely called even if
    the benchmark isn't complete.
    """
    benchmark = make_benchmark()
    benchmark.consolidate()


@main.command()
def generate_plots() -> None:
    """
    Generate PI plots.

    Make sure that the data is consolidated first: either by letting the
    benchmark run to completion, or by using the `consolidate` command.
    """
    benchmark = make_benchmark()
    benchmark._results = pd.read_csv(OUTPUT_DIR_PATH / "benchmark.csv")
    everything = product(
        benchmark._problems.keys(),
        benchmark._performance_indicators,
    )
    if not os.path.isdir(PLOTS_OUTPUT_DIR_PATH):
        os.mkdir(PLOTS_OUTPUT_DIR_PATH)
    for pn, pi in everything:
        print(f"Generating plot for problem '{pn}' and PI '{pi}'")
        try:
            grid = plot_performance_indicators(
                benchmark,
                row="algorithm",
                performance_indicators=[pi],
                problems=[pn],
            )
            grid.savefig(PLOTS_OUTPUT_DIR_PATH / f"{pn}.{pi}.jpg")
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(
                f"Error generating plot for problem '{pn}' and PI '{pi}':", e
            )


def make_benchmark() -> Benchmark:
    """Defines the benchmark"""
    # The noisy problems and their names
    noisy_problems = {
        "crs_8": make_c_region_simulator(8),
        "crs_8_1000": make_c_region_simulator(8, 1000),
        "pd_5": make_pendulum_cart(5),
        "zdt1_30": make_zdt1(30, 0.2),
    }
    # The knn-wrapped problems descriptions
    knn_problems = {
        f"{name}_k{k}": {
            "problem": KNNAvg(
                distance_weight_type="squared",
                max_distance=0.5,
                n_neighbors=k,
                problem=problem,
            )
        }
        for name, problem in noisy_problems.items()
        for k in [10, 25, 50, 100, 1000]
    }
    # The resample-wrapped problems descriptions
    avg_problems = {
        f"{name}_a{n}": {
            "evaluator": EvaluationPenaltyEvaluator("times", n),
            "problem": ResampleAverage(
                problem=problem,
                n_evaluations=n,
            ),
        }
        for name, problem in noisy_problems.items()
        for n in [10, 100]
    }
    # All the problem descriptions
    problems = {**knn_problems, **avg_problems}

    # Add additional problem data
    for name, problem in problems.items():
        problem["df_n_evals"] = 10
        if name.startswith("crs_8"):
            problem["hv_ref_point"] = np.array([100.0, 10.0])
        elif name.startswith("pd_5"):
            problem["hv_ref_point"] = np.array([10.0, 10.0])
        elif name.startswith("zdt1_30"):
            problem["hv_ref_point"] = np.array([10.0, 10.0])
        pf = problem["problem"].ground_problem().pareto_front()
        if not (pf is None):
            problem["pareto_front"] = pf

    # All algorithm descriptions
    algorithms = {
        f"nsga2_p{pop_size}": {
            "algorithm": NSGA2(pop_size=pop_size),
            "termination": MultiObjectiveDefaultTermination(
                n_max_evals=100_000,
            ),
        }
        for pop_size in [10, 20, 50]
    }

    return Benchmark(
        algorithms=algorithms,
        max_retry=10,
        n_runs=30,
        output_dir_path=OUTPUT_DIR_PATH,
        performance_indicators=["df", "hv", "igd"],
        problems=problems,
    )


def make_c_region_simulator(
    n_dimensions: int, n_threads=1
) -> CRegionSimulatorProblem:
    """
    Creates a `CRegionSimulatorProblem` wrapped inside a `nmoo.WrappedProblem`.

    Args:
        n_dimensions (int): Number of dimension for the `c_region_simulator`
            problem. This results in an actual search space of
            `6 * n_dimensions` dimensions.
        n_threads (int): Number of threads each `c_region_simulator` process is
            allowed to use. Defaults to 1.
    """
    problem = CRegionSimulatorProblem(
        c_region_simulator_path=C_REGION_SIMULATOR_PATH,
        n_dimensions=n_dimensions,
        n_workers=1,
        batch_size=20,
        A=np.array(
            [
                1.0105552342545365,
                0.10102185171959671,
                0.01010218517195967,
                1.0105552342545365,
            ]
        ),
        B=np.array(
            [
                0.005034383804730408,
                0.10052851662633117,
            ]
        ),
        c=1.0,
        default_vD=1.5,
        K=np.array([-1.0, -1.0]),
        kappaA=10,
        kappaD=10,
        modeA=3,
        modeD=3,
        neg=np.array([1.0, 1.0]),
        nS=n_threads,
        overline_vA=1,
        overline_vD=1.5,
        pos=np.array([4.0, 0.0]),
        sigmasq=0.01,
        T=200,
        threadCount=8,
        underline_vA=0,
        underline_vD=1.0,
        x0max=np.array([1.0, 1.0]),
        x0min=np.array([-1.0, -1.0]),
        cA1_min=np.array([-2.0] * n_dimensions),
        cA1_max=np.array([2.0] * n_dimensions),
        cA2_min=np.array([-2.0] * n_dimensions),
        cA2_max=np.array([2.0] * n_dimensions),
        cD1_min=np.array([-2.0] * n_dimensions),
        cD1_max=np.array([2.0] * n_dimensions),
        cD2_min=np.array([-2.0] * n_dimensions),
        cD2_max=np.array([2.0] * n_dimensions),
        vA_min=np.array([0.0] * n_dimensions),
        vA_max=np.array([30.0] * n_dimensions),
        vD_min=np.array([1.0] * n_dimensions),
        vD_max=np.array([30.0] * n_dimensions),
    )
    return WrappedProblem(problem)


def make_pendulum_cart(n_dimensions: int) -> PendulumCartProblem:
    """
    Creates a `PendulumCartProblem` wrapped inside a `nmoo.WrappedProblem`.

    Args:
        n_dimensions (int): Number of dimension for the `pendulum_cart`
            problem. This results in an actual search space of
            `2 * n_dimensions` dimensions.
    """
    problem = PendulumCartProblem(n_dimensions=n_dimensions, n_workers=-1)
    return WrappedProblem(problem)


def make_zdt1(n_var: int, noise: float) -> ZDT1:
    """Creates a noisy ZDT1 problem with Gaussian noise."""
    return GaussianNoise(
        WrappedProblem(ZDT1(n_var=n_var)),
        {"F": (np.array([0, 0]), noise * np.eye(2, dtype=float))},
    )


@main.command()
@click.option("--n-jobs", default=60)
def run(n_jobs: int) -> None:
    """Run the benchmark. It will automatically restart if it crashes."""
    restart = True
    while restart:
        try:
            benchmark = make_benchmark()
            benchmark.run(n_jobs, n_jobs, verbose=50)
        except KeyboardInterrupt:
            restart = False
        except Exception as e:
            print()
            print("===============================")
            print("BENCHMARK CRASHED... RESTARTING")
            print(e)
            print("===============================")
            print()
        else:
            restart = False


@main.command()
def tally() -> None:
    """
    Prints how many pairs have been successfully run against against how many
    pairs there are in total. It is safe to run this from a different shell
    while the benchmark is running.
    """
    pairs = make_benchmark()._all_pairs()
    n = sum(
        map(
            lambda p: int((OUTPUT_DIR_PATH / p.result_filename()).is_file()),
            pairs,
        )
    )
    print(f"{n}/{len(pairs)}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG,
    )
    main()
