from pathlib import Path

import nmoo
import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from src.c_region_simulator_problem import CRegionSimulatorProblem
from src.pendulum_cart_problem import PendulumCartProblem

from pymoo.config import Config
Config.show_compile_hint = False

C_REGION_SIMULATOR_PATH = (
    Path(__file__).absolute().parent / "src"
    / "c_region_simulator_problem"
    / "c_region_simulator_with_pipe"
)
print(C_REGION_SIMULATOR_PATH)

performance_indicators = ["dfp", "rghv", "rgigd"]

def make_benchmark() -> nmoo.Benchmark:
    """Defines the benchmark"""
    # The noisy problems and their names
    noisy_problems: dict[str, nmoo.WrappedProblem] = {
        # "crs_8": make_c_region_simulator(8),
        "crs_8_1000": make_c_region_simulator(8, 1000),
        "pd_5": make_pendulum_cart(5),
    }
    # The knn-wrapped problems descriptions
    knn_problems = {
        f"{name}_k{k}": {
            "problem": nmoo.KNNAvg(
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
    res_avg_problems = {
        f"{name}_R{n}": {
            "evaluator": nmoo.PenalizedEvaluator(multiplier=n),
            "problem": nmoo.ResampleAverage(
                problem=problem,
                n_evaluations=n,
            ),
        }
        for name, problem in noisy_problems.items()
        for n in [10, 100]
    }
    # All the problem descriptions
    problems = {**knn_problems, **res_avg_problems}

    # Add additional problem data
    for name, problem in problems.items():
        problem["df_n_evals"] = 10
        problem["rg_n_eval"] = 10
        if name.startswith("crs_8"):
            problem["hv_ref_point"] = np.array([100.0, 10.0])
        elif name.startswith("pd_5"):
            problem["hv_ref_point"] = np.array([10.0, 10.0])
        pf = problem["problem"].ground_problem().pareto_front()
        if not (pf is None):
            problem["pareto_front"] = pf

    # All algorithm descriptions
    algorithms = {
        f"nsga2_p{pop_size}": {
            "algorithm": NSGA2(pop_size=pop_size),
            "termination": MultiObjectiveDefaultTermination(
                n_max_evals=20_000,
            )
        }
        for pop_size in [20, 50]
    }

    return nmoo.Benchmark(
        output_dir_path="./casestudy-results",
        problems=problems,
        algorithms=algorithms,
        n_runs=30,
        seeds=list(np.arange(100, 10000, 100)),
        max_retry=10,
        performance_indicators=["dfp", "rghv", "rgigd"]
    )


def make_ar_benchmark() -> nmoo.Benchmark:
    """Defines the benchmark"""
    # The noisy problems and their names
    noisy_problems: dict[str, nmoo.WrappedProblem] = {
        # "crs_8": make_c_region_simulator(8),
        "crs_8_1000": make_c_region_simulator(8, 1000),
        "pd_5": make_pendulum_cart(5),
    }

    # The resample-wrapped problems descriptions
    noisy_problems: dict[str,dict] = {
        f"{name}": {
            "problem": problem
        }
        for name, problem in noisy_problems.items()
    }

    # Add additional problem data
    for name, problem in noisy_problems.items():
        problem["df_n_evals"] = 10
        problem["rg_n_eval"] = 10
        if name.startswith("crs_8"):
            problem["hv_ref_point"] = np.array([100.0, 10.0])
        elif name.startswith("pd_5"):
            problem["hv_ref_point"] = np.array([10.0, 10.0])
        pf = problem["problem"].ground_problem().pareto_front()
        if not (pf is None):
            problem["pareto_front"] = pf

    # All algorithm descriptions
    algorithms = {
        f"arnsga2_p{pop_size}": {
            "algorithm": nmoo.ARNSGA2(pop_size=pop_size, resampling_method="elite"),
            "termination": # get_termination("n_eval", 5000),
                MultiObjectiveDefaultTermination(
                    n_max_evals=20_000,
                ),
        }
        for pop_size in [20, 50]
    }

    return nmoo.Benchmark(
        output_dir_path="./casestudy-AR-results",
        problems=noisy_problems,
        algorithms=algorithms,
        max_retry=10,
        n_runs=30,
        seeds=list(np.arange(100, 10000, 100)),
        performance_indicators=["dfp", "rghv", "rgigd"]
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
    return nmoo.WrappedProblem(problem)


def make_pendulum_cart(n_dimensions: int) -> PendulumCartProblem:
    """
    Creates a `PendulumCartProblem` wrapped inside a `nmoo.WrappedProblem`.

    Args:
        n_dimensions (int): Number of dimension for the `pendulum_cart`
            problem. This results in an actual search space of
            `2 * n_dimensions` dimensions.
    """
    problem = PendulumCartProblem(n_dimensions=n_dimensions, n_workers=-1)
    return nmoo.WrappedProblem(problem)
