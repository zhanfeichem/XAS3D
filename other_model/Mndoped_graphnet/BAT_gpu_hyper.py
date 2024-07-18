import numpy as np
import os
import shutil


prefix_job="Mnep1000graphnet"
n_job=4
dir=os.path.dirname(__file__)  
dir_bat=os.path.join(dir,"bat")
if not (os.path.exists(dir_bat)):
    os.makedirs(dir_bat)

for i in range(n_job):

    iname=str(i)
    fbat=os.path.join(dir_bat,str(i)+".sh")
    with open(fbat,"w") as f:
        f.write(
    '''#!/bin/bash
# ========= Part 1 : Job Parameters ============
#SBATCH --partition=gpu
#SBATCH --qos=blnormal
#SBATCH --account=bldesign
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1''')
        f.write("\n")
        f.write("#SBATCH --job-name="+iname+prefix_job+"\n")
        f.write("#========== Part 2 : Job workload ============="+"\n")
        f.write("export PATH=$PATH:/scratchfs/heps/zhanf/parallel_fdmnes"+"\n")
        f.write("export PATH=/ihepfs/bldesign/user/zhanf/anaconda3/bin:$PATH" + "\n")
        f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ihepfs/bldesign/user/zhanf/anaconda3/lib" + "\n")
        f.write("source /ihepfs/bldesign/user/zhanf/anaconda3/bin/activate" + "\n")
        f.write("export MKL_THREADING_LAYER=GNU " + "\n")
        f.write("cd /ihepfs/bldesign/user/zhanf/pyg_1"+ "\n")
        f.write("python GraphNet_hyper_parallel.py  "+ iname +" > "+iname+".txt")
print("Finish sh file")

for i in range(n_job):
    fbat=os.path.join(dir_bat,str(i)+".sh")
    os.system("sbatch "+fbat )
print("Finish sbatch")

