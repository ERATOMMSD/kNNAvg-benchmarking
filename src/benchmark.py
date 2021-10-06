# Cédric's benchmark following the TestGen meeting of Oct. 6th
# https://docs.google.com/document/d/1u_aW9MJXIkrTs3DJbXoQ-dvrWZc0CW69YYCDh4W8_Z0/edit#heading=h.kto0noy3m605

# To run on a big boy

from pathlib import Path
import os

from nmoo.benchmark import Benchmark
from nmoo.denoisers import KNNAvg, ResampleAverage
from nmoo.evaluators import EvaluationPenaltyEvaluator
from nmoo.noises import GaussianNoise
from nmoo.wrapped_problem import WrappedProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi.zdt import ZDT1
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
import numpy as np

from c_region_simulator import CRegionSimulator
from pendulum_cart import PendulumCart

C_REGION_SIMULATOR_PATH = (
    Path(__file__).absolute().parent
    / "c_region_simulator"
    / "c_region_simulator_with_pipe"
)
OUTPUT_DIR_PATH = Path(__file__).absolute().parent / ".." / "out"


def make_c_region_simulator(n_dimensions: int) -> CRegionSimulator:
    """Creates a CRegionSimulator problem."""
    problem = CRegionSimulator(
        c_region_simulator_path=C_REGION_SIMULATOR_PATH,
        n_dimensions=n_dimensions,
        n_workers=-1,
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
        nS=1,
        overline_vA=1,
        overline_vD=1.5,
        pos=np.array([4.0, 0.0]),
        sigmasq=0.01,
        T=200,
        threadCount=1,
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


def make_pendulum_cart(n_dimensions: int) -> PendulumCart:
    """Creates a PendulumCart problem."""
    problem = PendulumCart(n_dimensions=n_dimensions, n_workers=-1)
    return WrappedProblem(problem)


def make_zdt1(n_var: int, noise: float) -> ZDT1:
    """Creates a noisy ZDT1 problem."""
    return GaussianNoise(
        WrappedProblem(ZDT1(n_var=n_var)),
        {"F": (np.array([0, 0]), noise * np.eye(2, dtype=float))},
    )


def main():
    # The noisy problems and their names
    noisy_problems = {
        "crs_8": make_c_region_simulator(8),
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

    # All algorithm descriptions
    algorithms = {
        f"nsga2_p{pop_size}": {
            "algorithm": NSGA2(pop_size=pop_size),
            "save_history": False,
            "termination": MultiObjectiveDefaultTermination(
                n_max_evals=100_000,
            )
        }
        for pop_size in [10, 20, 50]
    }

    # Benchmark definition and execution
    if not os.path.isdir(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)
    benchmark = Benchmark(
        algorithms=algorithms,
        n_runs=30,
        output_dir_path=OUTPUT_DIR_PATH,
        performance_indicators=[],
        problems=problems,
    )
    benchmark.run()


if __name__ == "__main__":
    main()