"""
CRegionSimulator call wrapped in a pymoo.Problem
"""

import subprocess
from pathlib import Path
from typing import Any, Union

import numpy as np
from pymoo.model.problem import Problem


class CRegionSimulator(Problem):

    _c_region_simulator_path: str
    _n_dimensions: int

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
        def _fmt(v: Any) -> str:
            """
            Formats value to a string that can be accepted as a command line
            argument for `c_region_simulator`.
            """
            if isinstance(v, np.ndarray):
                np.array2string(
                    v,
                    formatter={
                        "float_kind": lambda y: "%.1f" % y,
                    },
                    separator=" ",
                )[1:-1]
            if isinstance(v, bool):
                return "1" if v else "0"
            return str(v)

        cA1 = x[: self._n_dimensions]
        cA2 = x[self._n_dimensions : 2 * self._n_dimensions]
        cD1 = x[2 * self._n_dimensions : 3 * self._n_dimensions]
        cD2 = x[3 * self._n_dimensions : 4 * self._n_dimensions]
        vA = x[4 * self._n_dimensions : 5 * self._n_dimensions]
        vD = x[5 * self._n_dimensions :]

        args = [
                self._c_region_simulator_path,
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
                f"-cA1={_fmt(cA1)}",
                f"-cA2={_fmt(cA2)}",
                f"-cD1={_fmt(cD1)}",
                f"-cD2={_fmt(cD2)}",
                f"-vA={_fmt(vA)}",
                f"-vD={_fmt(vD)}",
            ]

        result = subprocess.run(
            args=args,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "c_region_simulator exited with code",
                result.returncode,
                str(result),
            )

        out["F"] = np.fromstring(result.stdout, dtype=float, sep=" ")

    def __init__(
        self,
        c_region_simulator_path: Union[Path, str],
        n_dimensions: int,
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
            elementwise_evaluation=True,
            **kwargs,
        )

        self._c_region_simulator_path = str(c_region_simulator_path)
        self._n_dimensions = n_dimensions
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


if __name__ == "__main__":
    import os

    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.factory import get_termination
    from pymoo.optimize import minimize

    c_region_simulator_path = os.path.abspath(
        __file__
        + "/../../submodules/controllerTesting/controller/CRegionSimulator/c_region_simulator"
    )

    n_dimensions = 8

    problem = CRegionSimulator(
        c_region_simulator_path=c_region_simulator_path,
        n_dimensions=n_dimensions,
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
        vD_min=np.array([0.0] * n_dimensions),
        vD_max=np.array([30.0] * n_dimensions),
    )

    algorithm = algorithm = NSGA2()

    results = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 40),
        # seed=1,
        save_history=True,
        verbose=True,
    )

    print(results.X)
    print(results.F)