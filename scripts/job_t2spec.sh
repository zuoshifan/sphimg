#!/bin/bash
#PBS -l nodes=30:ppn=8
#PBS -q batch
#PBS -r n
#PBS -m abe
#PBS -l walltime=07:00:00
#PBS -N cylinder_spec2

source /home/jrs65/pythonsetupbatch.sh

cd /home/jrs65/code/cylinder_simulation/scripts
export OMP_NUM_THREADS=8
mpirun -np 30 -npernode 1 python makecov_mpi.py /scratch/jrs65/cylinder/telescope2 ev2 &> /scratch/jrs65/cylinder/t2spec_ev2.out