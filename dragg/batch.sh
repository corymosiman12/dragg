#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=aipi0122@colorado.edu
#SBATCH --job-name="test"
#SBATCH --mail-type=ALL

module purge
source /curc/sw/anaconda3/2019.07/bin/activate
conda activate dragg

redis-server --daemonize yes
python -u main.py
