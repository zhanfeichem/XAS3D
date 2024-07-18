# XAS3D
We construct a 3D graph neural network model XAS3D to simulate XANES. It turns to be faster than the traditional XANES fitting method when we combine the simulation model and XANES optimization algorithm to fit the 3D structure of the given system. IHEP BSRF Zhaohf Group

example_Fe3O4:    python Fit.py
example_Mndoped:  python Fit_slurm.py

First, copy the dataset from the "datasets" folder to the "example" folder, or change the dataset file location in the code.
Then run "python Fit.py"

Installation Overview
conda create -n pyg_pl python==3.9
source /scratchfs/heps/zhanf/miniconda3/bin/activate pyg_pl
#
pip install pytorch-lightning==1.8 -i https://mirrors.ustc.edu.cn/pypi/web/simple
pip install tqdm
#PYG installation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install torch-sparse torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html 
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html   
pip install torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html 
pip install torchmetrics==0.7
#Fitting related package
pip install sympy
pip install nlopt
pip install pymatgen
pip install frechetdist
