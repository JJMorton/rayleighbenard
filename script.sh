#!/bin/bash
#SBATCH --export=ALL # export all environment variables to the batch job
#SBATCH -D . # set working directory to .
#SBATCH -p pq # submit to the parallel queue (pq for jobs, ptq for testing)
#SBATCH --time=72:00:00 # maximum walltime for the job (7d pq, 1h ptq)
#SBATCH -A Research_Project-127864 # research project to submit under
#SBATCH --nodes=8 # specify number of nodes
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=XXXXX@exeter.ac.uk # email address

# Unload any modules that are currently loaded
module purge

# Load modules
module load DEDALUS/2.1810-foss-2016a

sleep 10

NUMBEROFNODES=8
NUMPROCS=128

# Name of the executable which is used by the mpirun command
MYBIN1=3D_simulation.py
MYBIN2=merge.py
MYBIN3=interpolate.py

# Set the working directory. $SLURM_SUBMIT_DIR is the directory the job was submitted from but you could specify a different directory here if required
WORKDIR=$SLURM_SUBMIT_DIR

# Change to working directory and print a list of which nodes are being used
cd $WORKDIR || exec echo "Cannot cd to $WORKDIR"

mpiexec -np $NUMPROCS python3 $MYBIN1 ../analysis > log_sim
wait
mpiexec -np $NUMPROCS python3 $MYBIN2 ../analysis > log_merge
wait
mpiexec -np $NUMPROCS python3 $MYBIN3 ../analysis > log_interp
