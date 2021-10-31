# kNN averaging benchmarking

## Paper

*Trust your Neighbours: kNN-Averaging to Reduce Noise in Search-Based
Approaches*, Stefan Klikovits, Cédric Ho Thanh, Ahmet Cetinkaya, and Paolo
Arcaini.

## Structure of this repository

* `src/`: source code:
    * `src/benchmark.py`: main script containing the benchmark definition; run
      `python3 src/benchmark.py --help` for more information;
    * `src/c_region_simulator_problem/`: pymoo wrapper for the
      `c_region_simulator` problem; note that
      `src/c_region_simulator_problem/c_region_simulator_with_pipe` points to
      `submodules/controllerTesting/controller/CRegionSimulatorWithPipe/c_region_simulator_with_pipe`
      which is a binary you may need to recompile depending on your OS;
    * `src/pendulum_cart_problem/`:  pymoo wrapper for the `pendulum_cart`
      problem;
* `results/benchmarkN`: information and results regarding benchmark `N`; refer
  to `results/benchmarkN/README.md` for more details;
* `submodules/controllerTesting/`: dependency for `c_region_simulator` and
  `pendulum_cart`;

## How to run the benchmark

```sh
python -m nmoo run src.benchmark:make_benchmark
```

Refer to
```sh
python -m nmoo --help
python -m nmoo run --help
```
for more information.