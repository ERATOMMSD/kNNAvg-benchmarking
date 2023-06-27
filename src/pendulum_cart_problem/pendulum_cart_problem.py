"""
Pendulum cart problem wrapped inside a pymoo.Problem
"""
from math import ceil
from pathlib import Path
from typing import List, Tuple
import sys

from joblib import delayed, Parallel
from pymoo.core.problem import Problem
import numpy as np

sys.path.append(str(Path(__file__).absolute().parent) + "/")

from pcsim_old import pcsim
from loguru import logger as logging


class PendulumCartProblem(Problem):
    """
    Pendulum cart problem wrapped inside a pymoo.Problem.

    `pcsim` expects two input variables: `ps` and `pu`, each of n_dimensions
    `n_dimensions`. The underlying `pymoo.Problem` is initialized with `n_var =
    2 * n_dimensions`. In `_evaluate`, the input variable `x` (of shape `(N, 2
    * n_dimensions)`), is split into `N` `(ps, pu)` pairs (in that order).
    """

    _batch_size: int
    _n_workers: int

    _n_dimensions: int
    _noise_coefficient: float

    def __init__(
        self,
        n_dimensions: int,
        noise_coefficient: float = 0.001,
        ps_range: Tuple[float, float] = (0.0, 1.0),
        pu_range: Tuple[float, float] = (-20.0, 20.0),
        batch_size: int = 20,
        n_workers: int = 3,
        **kwargs,
    ):
        x_range = np.array(
            ([ps_range] * n_dimensions) + ([pu_range] * n_dimensions)
        )
        xl, xu = x_range.T
        super().__init__(
            n_var=2 * n_dimensions,
            n_obj=2,
            xl=xl,
            xu=xu,
            **kwargs,
        )
        self._batch_size = batch_size
        self._n_dimensions = n_dimensions
        self._n_workers = n_workers
        self._noise_coefficient = noise_coefficient

    def _evaluate(self, x, out, *args, **kwargs):
        def _run(y: List[np.ndarray]) -> np.ndarray:
            logging.debug("_run batch")

            results = [
                pcsim(
                    self._noise_coefficient,
                    w[: self._n_dimensions],
                    w[self._n_dimensions :],
                )
                for w in y
            ]
            return np.array(results)

        batches = np.array_split(x, ceil(len(x) / self._batch_size))
        jobs = [delayed(_run)(b) for b in batches]
        executor = Parallel(n_jobs=self._n_workers)
        logging.debug(f"About to trigger {len(jobs)} batches")
        results = executor(jobs)
        out["F"] = np.concatenate(results, axis=0)


if __name__ == "__main__":
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.factory import get_termination
    from pymoo.optimize import minimize

    n_dimensions = 5
    noise_coefficient = 0.001
    problem = PendulumCartProblem(
        batch_size=20,
        n_workers=-1,
        n_dimensions=n_dimensions,
        noise_coefficient=noise_coefficient,
    )

    algorithm = NSGA2()

    results = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 5),
        verbose=True,
    )

    # print(results.X)
    print(results.F)
