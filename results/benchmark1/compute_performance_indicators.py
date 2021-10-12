import os
from glob import glob
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.core.population import Population
from pymoo.factory import get_performance_indicator
from pymoo.problems.multi.zdt import ZDT1
from pymoo.util.optimum import filter_optimum

BENCHMARK_RESULT_PATH = Path(__file__).absolute().parent
PARETO_PATH = BENCHMARK_RESULT_PATH / "pareto"
if not os.path.isdir(PARETO_PATH):
    os.mkdir(PARETO_PATH)


def compute_pis(
    problem_name: str,
    algorithm_name: str,
    n_run: int,
) -> pd.DataFrame:
    pattern = (
        str(BENCHMARK_RESULT_PATH)
        + f"/{problem_name}.{algorithm_name}.{n_run}.1-*.npz"
    )
    path = glob(pattern)[0]

    data = np.load(path)
    pss = pareto_set_per_generation(data)
    dump_pareto_set_per_generation(
        pss, PARETO_PATH / f"{problem_name}.{algorithm_name}.{n_run}.pss.npz"
    )

    if problem_name.startswith("crs_8"):
        ref_point = np.array([100.0, 10.0])
        pf = pss[-1].get("F")
    elif problem_name.startswith("pd_5"):
        ref_point = np.array([10.0, 10.0])
        pf = pss[-1].get("F")
    elif problem_name.startswith("zdt1_30"):
        ref_point = np.array([10.0, 10.0])
        pf = ZDT1(n_var=30).pareto_front()

    pi_igd = get_performance_indicator("igd", pf)
    pi_hv = get_performance_indicator("hv", ref_point=ref_point)

    return pd.DataFrame(
        {
            "algorithm": algorithm_name,
            "problem": problem_name,
            "n_run": n_run,
            "n_gen": range(1, len(pss) + 1),
            "perf_igd": [pi_igd.do(ps.get("F")) for ps in pss],
            "perf_hv": [pi_hv.do(ps.get("F")) for ps in pss],
        }
    )


def compute_pis_safe(*args, **kwargs) -> pd.DataFrame:
    try:
        return compute_pis(*args, **kwargs)
    except:
        pass
    return pd.DataFrame()


def dump_pareto_set_per_generation(
    pss: List[Population], path: Union[Path, str]
) -> None:
    """
    In a structure simular to that generated by nmoo, dumps the pareto sets
    (`x` and `F` keys) for each generation (`_batch` key), given a list of
    populations.
    """
    all_x, all_F, all__batch = [], [], []
    for i, ps in enumerate(pss):
        all_x.append(ps.get("X"))
        all_F.append(ps.get("F"))
        all__batch.append(np.full(ps.get("X").shape[0], i + 1))
    np.savez_compressed(
        path,
        x=np.concatenate(all_x),
        F=np.concatenate(all_F),
        _batch=np.concatenate(all__batch),
    )


def main():
    benchmark_df = pd.read_csv(BENCHMARK_RESULT_PATH / "benchmark.csv")
    benchmark_df["timedelta"] = pd.to_timedelta(benchmark_df["timedelta"])
    jobs = [
        delayed(compute_pis_safe)(i, row)
        for i, row in enumerate(benchmark_df.iloc)
    ]
    result = Parallel(n_jobs=-1, verbose=50)(jobs)
    df = pd.concat(result, ignore_index=True)
    df.to_csv(BENCHMARK_RESULT_PATH / "benchmark_pis.csv", index=False)


def pareto_set_per_generation(data: np.lib.npyio.NpzFile) -> List[Population]:
    """
    Given an `NpzFile` containing keys `x`, `F`, `_batch`, returns the optimal
    population per generation (or batch). In other words, element `i` is the
    Pareto set at generation (or batch) `i`.
    """
    x, F, gen = data["x"], data["F"], data["_batch"]
    pop_all = Population.create()
    pss = []
    gmin, gmax = gen.min(), gen.max()
    for i in range(gmin, gmax + 1):
        idx = gen == i
        pop_i = Population.create(x[idx])
        pop_i.set(F=F[idx], feasible=np.full((x[idx].shape[0], 1), True))
        pop_all = Population.merge(pop_all, pop_i)
        pss.append(filter_optimum(pop_all))
    return pss


if __name__ == "__main__":
    main()
