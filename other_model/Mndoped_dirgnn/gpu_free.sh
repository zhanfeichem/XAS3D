#!/bin/bash
# ========= Part 1 : Job Parameters ============
#SBATCH --partition=gpu
#SBATCH --qos=blnormal
#SBATCH --account=bldesign
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=Mndir
#========== Part 2 : Job workload =============
export PATH=$PATH:/scratchfs/heps/zhanf/parallel_fdmnes
export PATH=/ihepfs/bldesign/user/zhanf/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ihepfs/bldesign/user/zhanf/anaconda3/lib
source /ihepfs/bldesign/user/zhanf/anaconda3/bin/activate
export MKL_THREADING_LAYER=GNU 
cd /ihepfs/bldesign/user/zhanf/pyg_1
python DIRGNN_once.py > 1once.txt
