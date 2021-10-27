# Benchmark 2

## Statement

* Problems:
  * **`pd_5`**: pendulum cart with `n_dimensions` = 5 (so the search space has
    dimension 10);
  * **`crs_8`**: `c_region_simulator` with `n_dimensions` = 8 (so the search
    space has dimension 48), no resampling;
  * **`crs_8_1000`**: `c_region_simulator` with `n_dimensions` = 8 (so the
    search space has dimension 48), number of resamplings (`nS`) = 1000;
  * **`zdt1_30`**: ZDT1 in a 30-dimensional search space + Gaussian noise;
* Denoisers:
  * **`kN`**: kNN averaging with `N` = 10, 25, 50, 100, 1000;
  * **`aN`**: resample averaging with number of resampling (`N`) = 10, 100;
* Algorithms:
  * **`nsga2_pN`**: NSGA2 with population size (`N`) = 10, 20, 50, maximum
    number of evaluations = 100000;
* Repetition for each problem-algorithm pair: 30.

## Results

* [Artifact link](). Contains:
  * `*.npz`: history files;
  * `*.csv`: summary files;
  * `*.pi.csv`: performance indicator files;
  * `*.pp.npz`: Pareto population files;
  * `*.gpp.npz`: global Pareto population files;
* `plots/`: Some plots, following this naming scheme:
  ```
  <problem name>_<denoiser name>.<performance indicator>.jpg
  ```

## Remarks

Ran on `nmoo v4`. Corrects the problems of benchmark 1.
