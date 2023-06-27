import args_dict as ar
import costs as co
import numpy as np
import pendulum_cart as pe
import mc
import numpy as np
import os
import os.path
import subprocess

def get_fullpath(filename):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, filename)

CPCSIMPATH = get_fullpath("cpcsim")

def pcsim(noiseCoefficient, ps, us):
    """noiseCoefficient: float (default 0.001)
       ps: np.ndarray (each entry should be between 0.0 and 1.0)
       us: np.ndarray (each entry should be between -20.0 and 20.0;
                       ps and us must have the same number of elements)

       pcsim returns a list of two floats representing objectives """

    ps_str = " ".join([f"{i:.15f}" for i in ps.tolist()])
    us_str = " ".join([f"{i:.15f}" for i in us.tolist()])
    output = subprocess.getoutput(f"{CPCSIMPATH} -noiseCoefficient=\"{noiseCoefficient:.15f}\" -ps=\"{ps_str}\" -us=\"{us_str}\"")
    return [float(o) for o in output.split(" ")]
