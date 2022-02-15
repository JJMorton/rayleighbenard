**Scripts for Rayleigh-Benard convection in 2.5D**

To run a simulation:
1. Create a directory for the analysis files to be saved in (e.g. `./analysis`)
2. Create `params.json` in said directory (there is an example at `./analysis/params.json`)
3. Run `python3 scripts/simulation.py <analysis dir>` or `mpiexec -n 2 -- python3 scripts/simulation.py <analysis dir>` to run the simulation with (for example) two threads
4. Merge the analysis files by running `python3 scripts/merge.py <analysis dir>`. You should end up with two files in the analysis directory: `state.h5` and `analysis.h5`
5. Create the plots by running `python3 scripts/plot.py <analysis_dir>`

The plots related to filtering also require the `interp.h5` file to be created by first running `python3 scripts/interpolate.py`.
The file `interp.h5` contains the velocity fields interpolated onto an evenly-spaced grid so that we can Fourier transform them.
