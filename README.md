# kNN averaging benchmarking

## Paper

*Trust your Neighbours: kNN-Averaging to Reduce Noise in Search-Based
Approaches*, Stefan Klikovits, Cédric Ho Thanh, Ahmet Cetinkaya, and Paolo
Arcaini.

## Structure of this repository

* `make_benchmark.py`: creates nmoo benchmark for synthetic problems. Use with `:make_benchmark`, `:make_ar_benchmark` or `:make_gpss_benchmark` 
* `utils.py`: helper functions to generate synthetic pymoo problems.
* `hv_refpoints.csv`: the reference points for the hypervolume calculation of the synthetic problems
* `make_casestudy.py`: creates nmoo benchmark for casestudy problems. Use with `:make_benchmark` or `:make_ar_benchmark`
* `src/`: source code:
    * `src/c_region_simulator_problem/`: pymoo wrapper for the
      `c_region_simulator` problem; note that
      `src/c_region_simulator_problem/c_region_simulator_with_pipe` points to
      `submodules/controllerTesting/controller/CRegionSimulatorWithPipe/c_region_simulator_with_pipe`
      which is a binary you may need to recompile depending on your OS;
    * `src/pendulum_cart_problem/`:  pymoo wrapper for the `pendulum_cart`
      problem;
* `submodules/controllerTesting/`: dependency for `c_region_simulator` and
  `pendulum_cart`;

## How to run the benchmark
Please refer to the [nmoo](https://github.com/altaris/noisy-moo) documentation for more info.
```sh
python -m nmoo run make_benchmark:make_benchmark
```

You might also refer to
```sh
python -m nmoo --help
python -m nmoo run --help
```
for more information.
