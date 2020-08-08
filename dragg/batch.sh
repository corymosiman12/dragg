#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=aipi0122@colorado.edu
#SBATCH --job-name="test"
#SBATCH --mail-type=ALL
#SBATCH --qos=general-compute

module purge
module load python

source /projects/$USER/software/anaconda/envs/dragg/bin/activate

python ./main.py
