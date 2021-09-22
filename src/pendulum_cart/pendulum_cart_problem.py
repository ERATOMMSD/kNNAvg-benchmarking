"""
Pendulum cart problem wrapped inside a pymoo.Problem
"""

from typing import Any, Dict, Union

import numpy as np
from pymoo.core.problem import Problem


# paramdic:
#     type
#     mp
#     mc
#     L
#     g
#     w

# simdic:
#     EMN
#     EMT
#     x0
#     uprightThreshold
#     umax
#     umin
#     xmax
#     xmin

# args:
#     threadCount
#     TOL
#     DELTA
#     nSMIN
#     nSMAX
#     nS
#     ps
#     us
#     try
#     printSimulationCount
#     printnS
#     twoObjectives
#     noiseCoefficient


class PendulumCart(Problem):
    """
    Pendulum cart problem wrapped inside a pymoo.Problem
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            n_var=...,
            n_obj=...,
            n_constr=...,
            xl=...,
            xu=...,
            **kwargs
        )
