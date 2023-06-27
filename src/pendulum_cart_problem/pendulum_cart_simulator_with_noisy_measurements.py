import animate as an
import args_dict as ar
import costs as co
import multiprocessing as mu
import numpy as np
import pendulum_cart as pe
import mc


paramdic = {
    "type": "nonlinear",
    "mp": 0.5,
    "mc": 0.5,
    "L": 1.4,
    "g": 10,
    "w": 1.0,
}

simdic = {
    "EMN": 100,
    "EMT": 10,
    "x0": np.matrix([[0], [0], [np.pi], [0]]),
    "uprightThreshold": 0.1,
    "umax": 20,
    "umin": -20,
    "xmax": 3,
    "xmin": -3,
}

args_dict = ar.ArgsDict(
    [
        "threadCount",
        "TOL",
        "DELTA",
        "nSMIN",
        "nSMAX",
        "nS",
        "ps",
        "us",
        "try",
        "printSimulationCount",
        "printnS",
        "twoObjectives",
        "noiseCoefficient",
    ]
)
simdic["ps"] = np.array(args_dict.get_double_list("ps"))
simdic["us"] = np.array(args_dict.get_double_list("us"))
paramdic["w"] = 1.0
if args_dict.args_dict["noiseCoefficient"] is not None:
    paramdic["w"] = args_dict.get_double("noiseCoefficient")

if args_dict.args_dict["try"] is not None:
    an.animate(
        pe.PendulumCart(paramdic).simulate_once(
            simdic, activate_closed_loop=True
        ),
        paramdic,
        simdic,
        "animations/standard_with_control_activation.mp4",
        save=False,
    )
else:
    simdic["threadCount"] = args_dict.get_int("threadCount")

    co.set_cost124(simdic)
    co.set_control_cost(simdic)

    def calculate_J_values(i):
        return pe.PendulumCart(paramdic).calculate_J_values(simdic)

    def compute_all_J_values(nS):
        pool_indices = list(range(nS))
        with mu.Pool(processes=simdic["threadCount"]) as pool:
            all_J_values = pool.map(calculate_J_values, pool_indices)
        return all_J_values

    def average_J_values(paramdic, simdic):
        all_J_values = compute_all_J_values(simdic["nS"])

        avg_J_values = [0 for J in simdic["Js"]]
        for i in range(simdic["nS"]):
            for j in range(len(simdic["Js"])):
                avg_J_values[j] = (
                    avg_J_values[j] * i + all_J_values[i][j]
                ) / (i + 1)

        return avg_J_values

    if args_dict.args_dict["nS"] is not None:
        simdic["nS"] = args_dict.get_int("nS")
    else:
        simdic["TOL"] = args_dict.get_double("TOL")
        simdic["DELTA"] = args_dict.get_double("DELTA")
        simdic["nSMIN"] = args_dict.get_int("nSMIN")
        simdic["nSMAX"] = args_dict.get_int("nSMAX")
        (M, total_M) = mc.find_simulation_count(simdic, compute_all_J_values)
        simdic["totalSimulationCount"] = total_M + M
        simdic["nS"] = M

    avg_J_values = average_J_values(paramdic, simdic)
    print("{:.15f}".format(avg_J_values[0]), end="")
    if args_dict.args_dict[
        "twoObjectives"
    ] is not None and args_dict.args_dict["twoObjectives"] == ["1"]:
        print(" {:.15f}".format(avg_J_values[1]), end="")
    if args_dict.args_dict[
        "printSimulationCount"
    ] is not None and args_dict.args_dict["printSimulationCount"] == ["1"]:
        print(" {}".format(simdic["totalSimulationCount"]), end="")
    if args_dict.args_dict["printnS"] is not None and args_dict.args_dict[
        "printnS"
    ] == ["1"]:
        print(" {}".format(simdic["nS"]), end="")
    print("")
