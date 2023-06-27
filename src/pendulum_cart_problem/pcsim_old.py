import args_dict as ar
import costs as co
import numpy as np
import pendulum_cart as pe
import mc
import numpy as np


def pcsim(noiseCoefficient, ps, us):
    """noiseCoefficient: float (default 0.001?)
    ps: np.ndarray (each entry should be between 0.0 and 1.0)
    us: np.ndarray (each entry should be between -20.0 and 20.0;
                    ps and us must have the same number of elements)

    pcsim returns a list of two floats representing objectives"""

    paramdic = {
        "type": "nonlinear",
        "mp": 0.5,
        "mc": 0.5,
        "L": 1.4,
        "g": 10,
        "w": 1.0,
    }

    simdic = {
        "EMN": 1000,  # If simulation is slow, we can set to 100
        "EMT": 10,
        "x0": np.matrix([[0], [0], [np.pi], [0]]),
        "uprightThreshold": 0.1,
        "umax": 20,
        "umin": -20,
        "xmax": 3,
        "xmin": -3,
    }

    simdic["ps"] = ps
    simdic["us"] = us
    paramdic["w"] = noiseCoefficient

    co.set_cost124(simdic)
    co.set_control_cost(simdic)

    return pe.PendulumCart(paramdic).calculate_J_values(simdic)


# print(pcsim(0.001, np.array([0.1, 0.2, 0.1, 0.3, 0.5]), np.array([10.2, -17.2, 0.5, -5.6, 9.1])))
