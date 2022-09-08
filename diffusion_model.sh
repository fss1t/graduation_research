#!/bin/bash

#$ -l rt_AF=2
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8
module load openmpi/4.1.3
module load cuda/11.0/11.0.3
module load cudnn/8.3/8.3.3
module load nccl/2.8/2.8.4-1

cd /home/ace14678rn/SONY/diffusion-models
mpirun -N 8 python3 train.py