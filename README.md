**Scripts for Rayleigh-Benard convection in 2.5D**

To run a simulation:
1. Create a directory for the analysis files to be saved in (e.g. `./analysis`)
2. Create `params.json` in said directory (there is an example at `./analysis/params.json`)
3. Run `python scripts/simulation.py <analysis dir>` or `mpiexec -n 2 -- python scripts/simulation.py <analysis dir>` to run the simulation with (for example) two threads
4. Merge the analysis files by running `python scripts/merge.py <analysis dir>`

I am putting my plotting routines in `scripts/plot.py` (`plotting_old.py` is from when I was using a Jupyter notebook and is generally a bit of a mess)
