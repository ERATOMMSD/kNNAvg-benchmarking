# Installation

## `controllerTesting`

Requires [GSL and GSL CBLAS libraries](https://www.gnu.org/software/gsl/).
```
cd submodules/controllerTesting/controller/CRegionSimulator
gcc -o c_region_simulator *.c -lgsl -lgslcblas -lm -lpthread
```

Example invokation:
```
./c_region_simulator -nS="1" -T="500" -threadCount="1" -cATheta="0.785398 2.356194 3.926991 5.497787" -pos="0 0" -neg="1 2" -A="1.0105552342545365 0.10102185171959671 0.01010218517195967 1.0105552342545365" -B="0.005034383804730408 0.10052851662633117" -K="-1 -1" -c="1.0" -sigmasq="0.01" -kappaA="10" -overline_vA="10" -underline_vA="0" -modeA="1" -kappaD="10" -overline_vD="10" -default_vD="1.0" -underline_vD="1.0" -modeD="0" -vA="1 2 3 4" -printTrajectory=0 -x0min="1 1" -x0max="1 1"
```
