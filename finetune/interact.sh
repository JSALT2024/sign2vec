#!/bin/bash
#SBATCH --job-name=MIG
#SBATCH --time=1:00:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1

#
source /data/apps/go.sh
ml cuda
cd Training-scripts/CUDA
nvcc -o matgpu.x matrixmul.cu
./matgpu.x