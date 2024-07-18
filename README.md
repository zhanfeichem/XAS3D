# XAS3D
We construct a 3D graph neural network model XAS3D to simulate XANES. It turns to be faster than the traditional XANES fitting method when we combine the simulation model and XANES optimization algorithm to fit the 3D structure of the given system. IHEP BSRF Zhaohf Group
#
## Installation Overview
conda create -n pyg_pl python==3.9 </br>
source /scratchfs/heps/zhanf/miniconda3/bin/activate pyg_pl </br>
pip install pytorch-lightning==1.8 -i https://mirrors.ustc.edu.cn/pypi/web/simple </br>
pip install tqdm </br>
### PYG installation </br>
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple  </br>
pip install torch-sparse torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html  </br>
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html   </br>
pip install torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html  </br>
pip install torchmetrics==0.7 </br>
### Fitting related package </br>
pip install sympy </br>
pip install nlopt </br>
pip install pymatgen </br>
pip install frechetdist </br>
#
## Running
example_Fe3O4:    python Fit.py</br>
example_Mndoped:  python Fit.py</br>
First, copy the dataset from the "datasets" folder to the "example" folder, or change the dataset file location in the code.</br>
Then run "python Fit.py"</br>