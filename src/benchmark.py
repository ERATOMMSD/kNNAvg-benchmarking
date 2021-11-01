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
from joblib import Parallel, delayed
from nmoo.benchmark import Benchmark
from nmoo.denoisers import KNNAvg, ResampleAverage
from nmoo.evaluators import EvaluationPenaltyEvaluator
from nmoo.noises import GaussianNoise
from nmoo.plotting import plot_performance_indicators
from nmoo.utils.population import pareto_frontier_mask, population_list_to_dict
from nmoo.wrapped_problem import WrappedProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.problems.multi.zdt import ZDT1
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

try:
    from .c_region_simulator_problem import CRegionSimulatorProblem
    from .pendulum_cart_problem import PendulumCartProblem
except ImportError:
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
@click.option(
    "--n-jobs",
    default=-1,
    type=click.INT,
)
def compute_effective_global_pareto_populations(n_jobs: int) -> None:
    """
    Computes global Pareto population based on the denoised `F` values. Assumes
    that the top level history files have been patched with `patch_histories`.
    """

    def _do(self, problem_name: str, algorithm_name: str) -> None:
        """
        Computes the global Pareto population of a given problem-algorithm
        pair. See `compute_global_pareto_populations`.
        """
        egpp_path = (
            self._output_dir_path / f"{problem_name}.{algorithm_name}.egpp.npz"
        )
        if egpp_path.is_file():
            # Effective global Pareto population has already been calculated
            return
        logging.debug(
            "Computing global Pareto population for pair [%s - %s]",
            problem_name,
            algorithm_name,
        )
        populations = []
        for n_run in range(1, self._n_runs):
            path = (
                self._output_dir_path
                / f"{problem_name}.{algorithm_name}.{n_run}.pp.npz.denoised.npz"
            )
            if not path.exists():
                logging.debug(
                    "File %s does not exist. The corresponding pair most "
                    "likely hasn't finished or failed",
                    path,
                )
                continue
            data = np.load(path)
            population = Population.create(data["X"])
            population.set(
                F=data["F"],
                feasible=np.full((data["X"].shape[0], 1), True),
            )
            populations.append(population)

        # Dump eGPP
        merged = population_list_to_dict(populations)
        mask = pareto_frontier_mask(merged["F"])
        np.savez_compressed(
            egpp_path,
            **{k: v[mask] for k, v in merged.items()},
        )

    benchmark = make_benchmark()
    pa_pairs = product(
        benchmark._problems.keys(),
        benchmark._algorithms.keys(),
    )
    executor = Parallel(n_jobs=n_jobs, verbose=50)
    executor(delayed(_do)(benchmark, an, pn) for an, pn in pa_pairs)


@main.command()
@click.option(
    "--n-jobs",
    default=-1,
    type=click.INT,
)
def denoise_pareto_populations(n_jobs: int) -> None:
    """
    Patches top level histories files by adding a `F_denoised` key, containing
    10 times sampled and averaged measurements of the problem on the values at
    key `X`.
    """

    def _patch(pair):
        pp_path = OUTPUT_DIR_PATH / pair.pareto_population_filename()
        ppp_path = OUTPUT_DIR_PATH / (
            pair.pareto_population_filename() + ".denoised.npz"
        )
        if ppp_path.is_file():
            return
        raw = np.load(pp_path, allow_pickle=True)
        pp = dict(raw)
        raw.close()
        base_problem = pair.problem_description["problem"].ground_problem()
        problem = ResampleAverage(base_problem, 10)
        pp["F"] = problem.evaluate(pp["X"])
        np.savez_compressed(ppp_path, **pp)

    logging.info("Denoising Pareto populations")
    benchmark = make_benchmark()
    executor = Parallel(n_jobs, verbose=50)
    executor(delayed(_patch)(p) for p in benchmark._all_pairs())


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
                legend=False,
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
        "crs_8_1000": make_c_region_simulator(8, nS=1000, n_threads=8),
        "pd_5": make_pendulum_cart(5),
        "zdt1_30": make_zdt1(30, 0.2),
    }
    # Baseline problems, with no denoiser
    baseline_problems = {
        name: {"problem": problem} for name, problem in noisy_problems.items()
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
    # problems = {**baseline_problems, **knn_problems, **avg_problems}
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
    n_dimensions: int, nS=1, n_threads=1
) -> CRegionSimulatorProblem:
    """
    Creates a `CRegionSimulatorProblem` wrapped inside a `nmoo.WrappedProblem`.

    Args:
        n_dimensions (int): Number of dimension for the `c_region_simulator`
            problem. This results in an actual search space of
            `6 * n_dimensions` dimensions.
        nS (int): Number of repetitions of the simulation (see
            c_region_simulator doc). Defaults to 1.
        n_threads (int): Number of threads each `c_region_simulator` process is
            allowed to use. Defaults to 8.
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
        nS=nS,
        overline_vA=1,
        overline_vD=1.5,
        pos=np.array([4.0, 0.0]),
        sigmasq=0.01,
        T=200,
        threadCount=n_threads,
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


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG,
    )
    main()
