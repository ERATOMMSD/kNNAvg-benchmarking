# Benchmark 1

## Statement

* Problems:
  * **`pd_5`**: pendulum cart with `n_dimensions` = 5 (so the search space has
    dimension
    10);
  * **`crs_8`**: `c_region_simulator` with `n_dimensions` = 8 (so the search
    space
    has dimension 48), no resampling (`nS` = 1);
  * **`zdt1_30`**: ZDT1 in a 30-dimensional search space + Gaussian noise;
* Denoisers:
  * **`kN`**: kNN averaging with `N` = 10, 25, 50, 100, 1000;
  * **`aN`**: resample averaging with number of resampling (`N`) = 10, 100;
* Algorithms:
  * **`nsga2_pN`**: NSGA2 with population size (`N`) = 10, 20, 50, maximum
    number
    of evaluations = 100000;
* Repetition for each problem-algorithm pair: 30.

## Results

* [Artifact link](). Contains all the `.npz` history files.
* `benchmark.csv`: Benchmark summary without performance indicators.
* `benchmark.csv`: Benchmark summary with IGD and hypervolume performance
  indicators computed using `compute_performance_indicators.py`.

## Remarks

This benchmark suffered a few problems:
* no performance indicator was specified;
* it ran on `nmoo v3.8.1`, which did not compute Pareto populations;
* `save_history` was set to `False`, which means that only the final generation
  of a algorithm-problem pair is represented in the final `benchmark.csv`.
