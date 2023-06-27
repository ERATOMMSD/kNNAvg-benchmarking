import nmoo
from pymoo.factory import get_problem, get_termination, get_algorithm
import numpy as np
import itertools

from pymoo.config import Config
Config.show_compile_hint = False

import utils

problem_label_sets = {
    "zdt": [f"ZDT{problem_idx + 1}" for problem_idx in range(6)],
    "mw": [f"MW{problem_idx + 1}" for problem_idx in range(14)],
    "ctp": [f"CTP{problem_idx + 1}" for problem_idx in range(8)],
    "omni": [f"OmniTest_{problem_idx + 1}" for problem_idx in range(1, 5)],
    "sympart": ["sympart", "sympartrotated"],
    "other": ["bnh", "kursawe", "osy", "tnk", "truss2d", "welded_beam"],
}

all_problem_labels = [
    *problem_label_sets["zdt"],
    *problem_label_sets["mw"],
    *problem_label_sets["ctp"],
    *problem_label_sets["omni"],
    *problem_label_sets["sympart"],
    *problem_label_sets["other"],
]
print(all_problem_labels)

performance_indicators = ["dfp", "ghv", "gigd"]

def get_problems(selector=None):
    ###############
    # Problems
    ###############

    problem_labels = problem_label_sets.get(selector.lower(), all_problem_labels)

    # if selector.lower() in problem_label_sets:
    #     print("Running Problem set", selector, f". Running {len(problem_labels)} problems.")
    # else:
    #     print("Selector", selector.lower(), f"is not known. Running all {len(problem_labels)} problems.")

    pymoo_problems: list = [utils.label_to_problem(lab) for lab in problem_labels]
    wrapped_problems: list[nmoo.WrappedProblem] = [nmoo.WrappedProblem(prob, name=label) for prob, label in zip(pymoo_problems, problem_labels)]

    #########
    # Noise
    #########

    noise_levels = [0.1, 0.25, 0.5]     # CONFIG
    noisy_problems: list[nmoo.GaussianNoise] = []
    for wrapped, L, in itertools.product(wrapped_problems, noise_levels):
        mean_F = np.zeros(wrapped.ground_problem().n_obj)
        cov_F = L * np.eye(wrapped.ground_problem().n_obj)

        parameters = {"F": (mean_F, cov_F)}

        # if wrapped.ground_problem().n_constr > 0:
        #     mean_G = np.zeros(wrapped.ground_problem().n_constr)
        #     cov_G = L * np.eye(wrapped.ground_problem().n_constr)
        #     parameters["G"] = (mean_G, cov_G)

        noisy_problem = nmoo.GaussianNoise(wrapped, parameters=parameters, name=f"{str(wrapped)}--L{L}")
        noisy_problems.append(noisy_problem)

    #############
    # Denoisers
    #############

    resample = [5, 10, 50, 100]     # CONFIG
    res_avg_denoisers = []
    for noisy_problem, r in itertools.product(noisy_problems, resample):
        name = f"{str(noisy_problem)}-R{r}"
        res_avg = nmoo.ResampleAverage(noisy_problem, n_evaluations=r, name=name)
        res_avg_denoisers.append(res_avg)


    neighbors = [5, 10, 20, 50, 100, 500]     # CONFIG
    MDs = [0.1, 0.25, 0.5, 1]     # CONFIG
    knn_avg_denoisers = []
    for noisy_problem, k, md in itertools.product(noisy_problems, neighbors, MDs):
        name = f"{str(noisy_problem)}-k{k}-MD{md}"
        knn_avg = nmoo.KNNAvg(noisy_problem, max_distance=md, n_neighbors=k, distance_weight_type="squared", name=name)
        knn_avg_denoisers.append(knn_avg)

    return noisy_problems, knn_avg_denoisers, res_avg_denoisers


def make_benchmark() -> nmoo.Benchmark:  # CONFIG
    N_RUNS: int = 40  # will be overridden
    output_dir_path: str = "./benchmark-results"
    selector = "all"

    noisy_problems, knn_avg_denoisers, res_avg_denoisers = get_problems(selector=selector)

    ##############
    # Algorithms
    ##############

    POP_SIZES = [10, 20, 50, 100]  # CONFIG
    algorithms = {f"nsga2_{P}": get_algorithm("nsga2", pop_size=P) for P in POP_SIZES}

    #############
    # Benchmark
    #############

    benchmark = nmoo.Benchmark(
            output_dir_path=output_dir_path,
            problems={
                **{str(noisy_problem): dict(
                    problem=noisy_problem,
                    hv_ref_point=utils.get_hv_refpoint(noisy_problem),
                    pareto_front=utils.get_pareto_front(noisy_problem)
                ) for noisy_problem in noisy_problems},
                **{str(knn_avg): dict(
                    problem=knn_avg,
                    hv_ref_point=utils.get_hv_refpoint(knn_avg),
                    pareto_front=utils.get_pareto_front(knn_avg)
                ) for knn_avg in knn_avg_denoisers},
                **{str(res_avg): dict(
                    problem=res_avg,
                    hv_ref_point=utils.get_hv_refpoint(res_avg),
                    pareto_front=utils.get_pareto_front(res_avg),
                    evaluator=nmoo.PenalizedEvaluator(multiplier=res_avg._n_evaluations)
                ) for res_avg in res_avg_denoisers}
            },
            algorithms={
                label: dict(
                    algorithm=algo,
                    termination=get_termination("n_eval", 5000)
                )
                for label, algo in algorithms.items()
            },
            n_runs=N_RUNS,
            seeds=list(np.arange(100, 10000, 100)),
            max_retry=3,
            performance_indicators=performance_indicators
        )
    return benchmark


def make_ar_benchmark() -> nmoo.Benchmark:  # CONFIG
    N_RUNS: int = 40  # will be overridden
    output_dir_path: str = "./AR-results"
    selector = "all"

    noisy_problems, knn_avg_denoisers, res_avg_denoisers = get_problems(selector=selector)

    ##############
    # Algorithms
    ##############

    POP_SIZES = [10, 20, 50, 100]  # CONFIG

    algorithms = {f"arnsga2_{P}_{M}": nmoo.ARNSGA2(pop_size=P, resampling_method=M)
                        for P, M in itertools.product(POP_SIZES, ["elite"])}


    #############
    # Benchmark
    #############

    benchmark = nmoo.Benchmark(
            output_dir_path=output_dir_path,
            problems={
                **{str(noisy_problem): dict(
                    problem=noisy_problem,
                    hv_ref_point=utils.get_hv_refpoint(noisy_problem),
                    pareto_front=utils.get_pareto_front(noisy_problem)
                ) for noisy_problem in noisy_problems},
            },
            algorithms={
                label: dict(
                    algorithm=algo,
                    termination=get_termination("n_eval", 5000)
                )
                for label, algo in algorithms.items()
            },
            n_runs=N_RUNS,
            seeds=list(np.arange(100, 10000, 100)),
            max_retry=3,
            performance_indicators=performance_indicators
        )

    return benchmark


def make_gpss_benchmark() -> nmoo.Benchmark:  # CONFIG
    N_RUNS: int = 40  # will be overridden
    output_dir_path: str = "./gpss-results"
    selector = "all"

    noisy_problems, knn_avg_denoisers, res_avg_denoisers = get_problems(selector=selector)

    gpss_denoisers = []

    monte_carlo_values = [2000, 4000]
    for noisy_problem, mc_value in itertools.product(noisy_problems, monte_carlo_values):
        name = f"{str(noisy_problem)}-GPSS-mc{mc_value}"
        means = np.zeros(noisy_problem.ground_problem().n_var + 2)
        stds = np.ones(noisy_problem.ground_problem().n_var + 2)
        gpss = nmoo.GPSS(noisy_problem, xi_prior_mean=means, xi_prior_std=stds, n_mc_samples=mc_value, xi_map_search_n_iter=50, name=name)
        gpss_denoisers.append(gpss)

    ##############
    # Algorithms
    ##############

    POP_SIZES = [10, 20, 50, 100]  # CONFIG
    algorithms = {f"nsga2_{P}": get_algorithm("nsga2", pop_size=P) for P in POP_SIZES}

    #############
    # Benchmark
    #############

    benchmark = nmoo.Benchmark(
            output_dir_path=output_dir_path,
            problems={
                **{str(gpss_problem): dict(
                    problem=gpss_problem,
                    hv_ref_point=utils.get_hv_refpoint(gpss_problem),
                    pareto_front=utils.get_pareto_front(gpss_problem)
                ) for gpss_problem in gpss_denoisers},
            },
            algorithms={
                label: dict(
                    algorithm=algo,
                    termination=get_termination("n_eval", 5000)
                )
                for label, algo in algorithms.items()
            },
            n_runs=N_RUNS,
            seeds=list(np.arange(100, 10000, 100)),
            max_retry=3,
            performance_indicators=performance_indicators
        )
    return benchmark
