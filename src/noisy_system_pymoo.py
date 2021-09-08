import itertools
import os
from typing import List

import pandas as pd
from pymoo.model.evaluator import Evaluator
from pymoo.model.problem import Problem
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

import nmoo
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2

from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

import logging
logger = logging.getLogger(__name__)

class Resampling_Fix_NEval_Evaluator(Evaluator):
    """An Evaluator that corrects the internal n_eval if the problem is a ResampleAverage denoiser!"""
    def eval(self,
             problem,
             pop,
             **kwargs):
        orig_ret = super().eval(problem, pop, **kwargs)

        # correct self.n_eval if necessary !!!!
        if isinstance(problem, nmoo.denoisers.ResampleAverage):
            self.n_eval += (problem._n_evaluations - 1) * len(orig_ret)
        return orig_ret


def run_problem(problem_name: str, problem: Problem, noise: float, total_evaluations: int) -> List[pd.DataFrame]:
    logger.info(f"{problem_name} -- Vars: {problem.n_var} -- Obj: {problem.n_obj} -- total_evaluations: {total_evaluations}")
    out_dir = f"./output/{problem_name}_Noise{noise}"
    pareto_front = problem.pareto_front()

    wrapped_problem = nmoo.WrappedProblem("WrappedProblem", problem)

    # TODO: Implement noise
    noise_dict = {"F": (np.zeros(problem.n_obj), np.eye(problem.n_obj) * noise)}
    if problem.n_constr > 0:
        noise_dict["G"] = (np.zeros(problem.n_constr), np.eye(problem.n_constr) * noise)
    noisy_problem = nmoo.noises.GaussianNoise(wrapped_problem, noise_dict)

    k_values = [5, 10, 20, 50, 100, 500, 1000]
    knn_options = dict(max_distance=1, distance_weight_type="squared")
    knn_avg_problems = {f"{k}NN-Avg": {
                "problem": nmoo.denoisers.KNNAvg(noisy_problem, n_neighbors=k, **knn_options),
                "pareto_front": pareto_front,
            } for k in k_values}

    resample_values = [1] # [1, 5, 10, 50, 100]
    resample_problems = {f"resample{r}": {
                "problem": nmoo.denoisers.ResampleAverage(noisy_problem, n_evaluations=r),
                "pareto_front": pareto_front,
            } for r in resample_values}

    # TODO: Here we install our denoisers !

    # Run Algorithms
    gensizes = [10, 20, 50, 100]
    algorithms = { f"nsga2_GS{gs}" : {
        "algorithm": NSGA2(pop_size=gs,eliminate_duplicates=True),
        # "termination": get_termination("n_eval", total_evaluations) if total_evaluations > 0 else None
        "termination": MultiObjectiveDefaultTermination(
                            x_tol=1e-8,
                            cv_tol=1e-6,
                            f_tol=0.0025,
                            nth_gen=5,
                            n_last=30,
                            n_max_gen=4000,
                            n_max_evals=400000
                        )
    } for gs in gensizes}

    # JMetal does HyperParam search
    # irace (based on statistical test)

    os.makedirs(out_dir, exist_ok=True)
    benchmark = nmoo.benchmark.Benchmark(
        output_dir_path=out_dir,
        problems={
            "noisy": {
                "problem": noisy_problem,
                "pareto_front": pareto_front,
            },
            # **knn_avg_problems,
            # **resample_problems,
        },
        algorithms=algorithms,
        n_runs=3,
    )

    results = benchmark.run(verbose=-1, n_jobs=4, dump_results=False, evaluator=Resampling_Fix_NEval_Evaluator)
    benchmark.final_results().to_csv(out_dir+"/n_gen_check.csv")
    print(problem_name)
    print(benchmark.final_results())

    return results
