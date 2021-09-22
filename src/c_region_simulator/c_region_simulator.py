"""
CRegionSimulator call wrapped in a pymoo.Problem
"""

from pathlib import Path
from typing import Any, Dict, Union
import subprocess

from joblib import delayed, Parallel

import numpy as np
from pymoo.core.problem import Problem

C_REGION_SIMULATOR_POOL: Dict[int, subprocess.Popen] = {}


class CRegionSimulator(Problem):
    """
    The inputs of `c_region_simulator` are the variables `cA1`, `cA2`, `cD1`,
    `cD2`, `vA`, and `vD`, each of dimension `_n_dimensions`. However, in
    pymoo, an individual is a vector of length `6 * _n_dimensions`. It
    represents the concatenation of `cA1`, `cA2`, `cD1`, `cD2`, `vA`, and `vD`
    in that order.
    """

    _batch_size: int
    _c_region_simulator_path: str
    _n_dimensions: int
    _n_workers: int

    _cA1_min: np.ndarray
    _cA1_max: np.ndarray
    _cA2_min: np.ndarray
    _cA2_max: np.ndarray
    _cD1_min: np.ndarray
    _cD1_max: np.ndarray
    _cD2_min: np.ndarray
    _cD2_max: np.ndarray
    _vA_min: np.ndarray
    _vA_max: np.ndarray
    _vD_min: np.ndarray
    _vD_max: np.ndarray

    _A: np.ndarray
    _B: np.ndarray
    _c: float
    # _cATheta: np.ndarray
    _default_vD: float
    _K: np.ndarray
    _kappaA: float
    _kappaD: float
    _modeA: float
    _modeD: float
    _neg: np.ndarray
    _nS: float
    _overline_vA: float
    _overline_vD: float
    _pos: np.ndarray
    _printTrajectory: bool
    _sigmasq: float
    _T: float
    _threadCount: int
    _underline_vA: float
    _underline_vD: float
    _x0max: np.ndarray
    _x0min: np.ndarray

    def _evaluate(self, x, out, *args, **kwargs):
        """
        The inputs of `c_region_simulator` are the variables `cA1`, `cA2`,
        `cD1`, `cD2`, `vA`, and `vD`, each of dimension `_n_dimensions`.
        However, pymoo provides a single input vector. It has length `6 *
        _n_dimensions`, and it is split to produce `cA1`, `cA2`, `cD1`, `cD2`,
        `vA`, and `vD` in that order.

        In reality, things are a bit more subtle: `_evaluate` is _batched_,
        meaning that argument `x` is a `(N, 6 * _n_dimensions)` array. This
        method thus iterates over the rows of `x`, splits each row in `6`, and
        make `N` calls to `c_region_simulator`.
        """

        def _fmt(v: Any) -> str:
            """
            Formats value to a string that can be accepted as a command line
            argument for `c_region_simulator`.
            """
            if isinstance(v, np.ndarray):
                return np.array2string(
                    v,
                    formatter={
                        "float_kind": lambda y: "%.1f" % y,
                    },
                    separator=" ",
                )[1:-1]
            if isinstance(v, bool):
                return "1" if v else "0"
            return str(v)

        def _run(
            y: np.ndarray,
            base_args: Dict[str, str],
            process: subprocess.Popen,
        ) -> np.ndarray:
            out_f = []
            for w in y:
                cA1 = w[: self._n_dimensions]
                cA2 = w[self._n_dimensions : 2 * self._n_dimensions]
                cD1 = w[2 * self._n_dimensions : 3 * self._n_dimensions]
                cD2 = w[3 * self._n_dimensions : 4 * self._n_dimensions]
                vA = w[4 * self._n_dimensions : 5 * self._n_dimensions]
                vD = w[5 * self._n_dimensions :]
                x_args = [
                    f"-cA1={_fmt(cA1)}",
                    f"-cA2={_fmt(cA2)}",
                    f"-cD1={_fmt(cD1)}",
                    f"-cD2={_fmt(cD2)}",
                    f"-vA={_fmt(vA)}",
                    f"-vD={_fmt(vD)}",
                ]
                process.stdin.write(" ".join(base_args + x_args) + "\n")
                process.stdin.flush()
                raw_f = process.stdout.readline().strip()
                out_f.append(np.fromstring(raw_f, dtype=float, sep=" "))
            return np.array(out_f)

        base_args = [
            f"-A={_fmt(self._A)}",
            f"-B={_fmt(self._B)}",
            f"-c={_fmt(self._c)}",
            # f"-cATheta={_fmt(self._cATheta)}",
            f"-default_vD={_fmt(self._default_vD)}",
            f"-K={_fmt(self._K)}",
            f"-kappaA={_fmt(self._kappaA)}",
            f"-kappaD={_fmt(self._kappaD)}",
            f"-modeA={_fmt(self._modeA)}",
            f"-modeD={_fmt(self._modeD)}",
            f"-neg={_fmt(self._neg)}",
            f"-nS={_fmt(self._nS)}",
            f"-overline_vA={_fmt(self._overline_vA)}",
            f"-overline_vD={_fmt(self._overline_vD)}",
            f"-pos={_fmt(self._pos)}",
            f"-printTrajectory={int(self._printTrajectory)}",
            f"-sigmasq={_fmt(self._sigmasq)}",
            f"-T={_fmt(self._T)}",
            f"-threadCount={_fmt(self._threadCount)}",
            f"-underline_vA={_fmt(self._underline_vA)}",
            f"-underline_vD={_fmt(self._underline_vD)}",
            f"-x0max={_fmt(self._x0max)}",
            f"-x0min={_fmt(self._x0min)}",
        ]

        jobs = [
            delayed(_run)(y, base_args, self._get_c_region_simulator_instance(i))
            for i, y in enumerate(np.array_split(x, self._batch_size))
        ]
        executor = Parallel(
            backend="threading",
            n_jobs=self._n_workers,
        )
        results = executor(jobs)
        out["F"] = np.concatenate(results, axis=0)

    def _get_c_region_simulator_instance(self, index: int) -> subprocess.Popen:
        """
        Creates or gets an existing Popen object connected to a
        `c_region_simulator_with_pipe` instance.
        """
        global C_REGION_SIMULATOR_POOL
        if index not in C_REGION_SIMULATOR_POOL:
            process = subprocess.Popen(
                [self._c_region_simulator_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )
            C_REGION_SIMULATOR_POOL[index] = process
        return C_REGION_SIMULATOR_POOL[index]

    def __init__(
        self,
        c_region_simulator_path: Union[Path, str],
        n_dimensions: int,
        n_workers: int = 3,
        batch_size: int = 10,
        *,
        A: np.ndarray,
        B: np.ndarray,
        c: float,
        # cATheta: np.ndarray,
        default_vD: float,
        K: np.ndarray,
        kappaA: float,
        kappaD: float,
        modeA: float,
        modeD: float,
        neg: np.ndarray,
        nS: float,
        overline_vA: float,
        overline_vD: float,
        pos: np.ndarray,
        printTrajectory: bool = False,
        sigmasq: float,
        T: float,
        threadCount: int = 8,
        underline_vA: float,
        underline_vD: float,
        x0max: np.ndarray,
        x0min: np.ndarray,
        cA1_min: np.ndarray,
        cA1_max: np.ndarray,
        cA2_min: np.ndarray,
        cA2_max: np.ndarray,
        cD1_min: np.ndarray,
        cD1_max: np.ndarray,
        cD2_min: np.ndarray,
        cD2_max: np.ndarray,
        vA_min: np.ndarray,
        vA_max: np.ndarray,
        vD_min: np.ndarray,
        vD_max: np.ndarray,
        **kwargs,
    ):
        """
        See also:
            https://pymoo.org/problems/index.html
        """
        xl = np.array(
            [
                cA1_min,
                cA2_min,
                cD1_min,
                cD2_min,
                vA_min,
                vD_min,
            ]
        ).flatten()
        xu = np.array(
            [
                cA1_max,
                cA2_max,
                cD1_max,
                cD2_max,
                vA_max,
                vD_max,
            ]
        ).flatten()
        super().__init__(
            n_var=n_dimensions * 6,
            n_obj=2,
            n_constr=0,
            xl=xl,
            xu=xu,
            **kwargs,
        )

        self._batch_size = batch_size
        self._c_region_simulator_path = str(c_region_simulator_path)
        self._n_dimensions = n_dimensions
        self._n_workers = n_workers

        self._cA1_min = cA1_min
        self._cA1_max = cA1_max
        self._cA2_min = cA2_min
        self._cA2_max = cA2_max
        self._cD1_min = cD1_min
        self._cD1_max = cD1_max
        self._cD2_min = cD2_min
        self._cD2_max = cD2_max
        self._vA_min = vA_min
        self._vA_max = vA_max
        self._vD_min = vD_min
        self._vD_max = vD_max

        self._A = A
        self._B = B
        self._c = c
        # self._cATheta = cATheta
        self._default_vD = default_vD
        self._K = K
        self._kappaA = kappaA
        self._kappaD = kappaD
        self._modeA = modeA
        self._modeD = modeD
        self._neg = neg
        self._nS = nS
        self._overline_vA = overline_vA
        self._overline_vD = overline_vD
        self._pos = pos
        self._printTrajectory = printTrajectory
        self._sigmasq = sigmasq
        self._T = T
        self._threadCount = threadCount
        self._underline_vA = underline_vA
        self._underline_vD = underline_vD
        self._x0max = x0max
        self._x0min = x0min


def main():
    from pathlib import Path

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.factory import get_termination
    from pymoo.optimize import minimize

    n_dimensions = 8
    c_region_simulator_path = (
        Path(__file__).absolute().parent / "c_region_simulator_with_pipe"
    )
    problem = CRegionSimulator(
        c_region_simulator_path=c_region_simulator_path,
        n_dimensions=n_dimensions,
        n_workers=-1,
        batch_size=10,
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
        nS=10000,
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

    algorithm = algorithm = NSGA2()

    results = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 5),
        seed=1,
        save_history=True,
        verbose=True,
    )

    # print(results.X)
    # print(results.F)

if __name__ == "__main__":
    main()