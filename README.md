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

## Yielded vs. effective Pareto front

`nmoo` calculated distance-based performance indicators (e.g. `igd`) based on
the _yielded_ Pareto front. The `F` values contained in the `.pp.npz` and
`.gpp.npz` files are all considered _yielded_, i.e. they are all noisy.

However, in the paper, statistics are based on the _effective_ Pareto
front. Effective values are denoised by resampling-averaging 10 times. This
means that everything in the `.pp.npz` need to be denoised and the `.gpp.npz`
files recalculated. The first step can be done using
```sh
python3 src/benchmark.py denoise-pareto-populations
```
this will create `.pp.npz.denoised.npz` files, one for each _yielded_
`.pp.npz`. After backup, you can replace the `.pp.npz` files running e.g.
```sh
rm *.pp.npz
rename "s/(.*)\.denoised\.npz/\$1/" *
```
to rename all `.pp.npz.denoised.npz` files to `.pp.npz`. Then, run
```sh
python -m nmoo run src.benchmark:make_benchmark
```
once more to recalculate the global Pareto populations, the PIs, and to
consolidate.