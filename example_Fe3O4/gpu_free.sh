#!/bin/bash
# ========= Part 1 : Job Parameters ============
#SBATCH --partition=gpu
#SBATCH --qos=blnormal
#SBATCH --account=bldesign
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=abs
#========== Part 2 : Job workload =============
#export PATH=$PATH:/scratchfs/heps/zhanf/parallel_fdmnes
export PATH=/ihepfs/bldesign/user/zhanf/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ihepfs/bldesign/user/zhanf/anaconda3/lib
source /ihepfs/bldesign/user/zhanf/anaconda3/bin/activate
export MKL_THREADING_LAYER=GNU
python Fit_F1.py  > 1.log
