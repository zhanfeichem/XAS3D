#!/bin/bash
# ========= Part 1 : Job Parameters ============
#SBATCH --partition=gpu
#SBATCH --qos=blnormal
#SBATCH --account=bldesign
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --job-name=Mnabs
#========== Part 2 : Job workload =============
#export PATH=$PATH:/scratchfs/heps/zhanf/parallel_fdmnes
export ENV=/scratchfs/heps/zhanf/miniconda3/envs/pyg_pl
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV/lib
export MKL_THREADING_LAYER=GNU
nvidia-smi -L > 1gpu.log
$ENV/bin/python Fit_slurm.py  > 1.log
#
