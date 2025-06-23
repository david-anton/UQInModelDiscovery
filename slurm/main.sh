#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --job-name=BMDisc
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force statisticalmdisc.sif app/.devcontainer/container.def

SCRIPT=main.py

srun singularity run \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1\
 --nv \
 --nvccli \
 statisticalmdisc.sif \
 python3 /home/davanton/StatisticalModelDiscovery/app/${SCRIPT}


