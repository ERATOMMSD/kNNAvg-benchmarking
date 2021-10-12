# Benchmark 1

## Statement

* Problems:
  * `pendulum`: n_var = 2 * dim, dim = 5);
  * `c_region_sim`: `n_var = 6 * dim`, `dim` = 8, `nS` = 1, 1000;
  * `ZDT1` (`n_var` = 30) + Gaussian noise;
* Denoisers: KNN = 10, 25, 50, 100, 1000; Avg = 10, 100;
* NSGA2: `pop_size` = 10, 20, 50; `n_max_eval` = 100_000;
* Repetition for each problem-algo pair: 30

## Results

[Artifact
link](https://drive.google.com/file/d/1P0ZTZEcxg8xFhBBBv_9yLaC7nlwWQt2G/view?usp=sharing)
Contains all `npz` files and the `benchmark.csv` summary. Since the Pareto
fronts have not been calculated (yet), `benchmark.csv` does not contain
metrics.

## Remark

Unfortunately, this benchmark suffered a few problems:
* no performance indicator was specified;
* it ran on `nmoo 3.8.1`, which didn't implement default Pareto sets anyways;
* `save_history` was set to `False`, which means that only the final generation
  of a algorithm-problem pair is represented in the final `benchmark.csv`.
