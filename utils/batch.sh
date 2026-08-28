#!/bin/bash
#PBS -N compare_metadata
#PBS -l select=1:ncpus=8:mem=32Gb:ngpus=1
#PBS -l walltime=0:30:00
#PBS -j oe

# 1. Change to the directory where your script is located


# 2. Activate your micromamba environment
micromamba activate dino_env
            
export PYTHONUNBUFFERED=1
source activate dino_env

python /home/nicolasg/dev/compare_metadata.py > /home/nicolasg/dev/compare_metadata.log 2>&1


