

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn.functional as F
###DIG
# from dig.threedgraph.dataset import QM93D
# from dig.threedgraph.method import SphereNet
# from dig.threedgraph.method import DimeNetPP,ComENet
# from dig.threedgraph.evaluation import ThreeDEvaluator
# from dig.threedgraph.method import run
###PYG
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
######
from scipy.interpolate import interp1d
import numpy as np
import json
import argparse
import os.path as osp
import time
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as 
##########output
#from pymatgen.io.cif import CifWriter
##########ComENet
from XAS3Dabs_posv1 import XAS3Dabs
##########ComENet  END
class Cu_train(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix="Fe3O4_7113_17_21_39_site"
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_train_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_train_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_train_y.npy", allow_pickle=True)
        # r_list = np.load(prefix+"_train_r.npy",allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(np.int)
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            # r = torch.tensor(r_list[i])
            tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class Cu_test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix="Fe3O4_7113_17_21_39_site"
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_test_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_test_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_test_y.npy", allow_pickle=True)
        # r_list = np.load(prefix + "_test_r.npy", allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(np.int)
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            # r=torch.tensor(r_list[i])
            tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
'''END THE DATASET DEFINATION'''



t0=time.time()
n_ep=200
n_ep_save=400
cutoff=8.0
n_block=3
npt=120;#out_channels=npt  other define x_pred
setting_batch_size=32;setting_vt_batch_size=32;
# setting_batch_size=64;setting_vt_batch_size=32;
setting_fmodel="Mn_cen_-0.3_0.3_R5.pt"
setting_ini_model=False
''' END SETTING '''


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:",device)
print("EPOACH NUMBER:",n_ep)
###load dataset
# dim_out=100#out 100 energy point
dataset = Cu_train("Fe3O4_7113_17_21_39_site_train")
# dataset=dataset.shuffle()#random
dataset_test = Cu_test("Fe3O4_7113_17_21_39_site_test")

# dataset_valid0 = test_site0("FeOOH_0.3_R5_valid_site0")
# dataset_valid1 = test_site1("FeOOH_0.3_R5_valid_site1")
# dataset_valid2 = test_site2("FeOOH_0.3_R5_valid_site2")
# dataset_valid3 = test_site3("FeOOH_0.3_R5_valid_site3")

# dataset_test = dataset_test.shuffle()

n_plot=4*10
# dataset_plot=dataset[-n_plot:]#dataset for testing show
dataset_plot=dataset[0:n_plot]#dataset for train effectshow
loader_plot = DataLoader(dataset_plot, batch_size=n_plot)#once plot all
it_plot=loader_plot._get_iterator()
batch_plot=next(it_plot)
batch_plot=batch_plot.to(device)

n_test=20*24#
dataset_test=dataset_test[0:n_test]#dataset for show
loader_test = DataLoader(dataset_test, batch_size=n_test)#loader_test = DataLoader(dataset_test, batch_size=1)
it_test=loader_test._get_iterator()
batch_test=next(it_test)
batch_test=batch_test.to(torch.device("cpu"))#CPU test calculation

n_valid=20*24#
dataset_valid=dataset_test[0:n_valid]#dataset for show
loader_valid = DataLoader(dataset_valid, batch_size=1)

# n_valid_site=400
# dataset_valid0=dataset_valid0[0:n_valid_site]#dataset for show
# loader0 = DataLoader(dataset_valid0, batch_size=1,shuffle=False)
# dataset_valid1=dataset_valid1[0:n_valid_site]#dataset for show
# loader1 = DataLoader(dataset_valid1, batch_size=1,shuffle=False)
# dataset_valid2=dataset_valid2[0:n_valid_site]#dataset for show
# loader2 = DataLoader(dataset_valid2, batch_size=1,shuffle=False)
# dataset_valid3=dataset_valid3[0:n_valid_site]#dataset for show
# loader3 = DataLoader(dataset_valid3, batch_size=1,shuffle=False)

n_train=180*24 #180*24
dataset_train=dataset[0:n_train]
loader_train = DataLoader(dataset_train, batch_size=1)
print("Finish dataset")


global Nf
Nf=1


def f(params,irun):
    global Nf
    Nf=Nf+1
    print("Nf:",Nf)
    num_layers=params['num_layers']
    hidden_channels=params['hidden_channels']
    middle_channels=params['middle_channels']
    num_output_layers=params['num_output_layers']
    num_radial=params['num_radial']
    num_spherical=params['num_spherical']

    # cutoff = 8.0,
    # num_layers = 4,
    # hidden_channels = 256,
    # middle_channels = 64,
    # out_channels = 1,
    # num_radial = 3,
    # num_spherical = 2,
    # num_output_layers = 3,

    print("CUTOFF:",cutoff)
    print("NBLOCK:",n_block)
    # model = ComENet(out_channels=dim_out,cutoff=cutoff,num_layers=n_block)
    model = XAS3Dabs(out_channels=npt,cutoff=cutoff
                      ,num_layers=num_layers,hidden_channels=hidden_channels,middle_channels=middle_channels
                      ,num_output_layers=num_output_layers,num_radial=num_radial,num_spherical=num_spherical)

    import torch




    loss_func = torch.nn.L1Loss()
    import time
    ta=time.time()
    tb=time.time()
    print("TIME SPEND:",tb-ta)
    import time
    import os
    import torch
    from torch.optim import Adam
    from torch_geometric.data import DataLoader
    import numpy as np
    from torch.autograd import grad
    #from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    epochs=n_ep
    lr=0.0005;weight_decay=0;lr_decay_step_size=50;lr_decay_factor=0.5;
    batch_size=setting_batch_size;vt_batch_size=setting_vt_batch_size;
    train_dataset=dataset_train;valid_dataset=dataset_valid;test_dataset=dataset_test;
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
    valid_loader_save = DataLoader(valid_dataset, 1, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    best_valid = float('inf')
    best_test = float('inf')

    save_loss=[]
    save_loss_valid=[]
    save_time=[]
    for epoch in range(1, epochs + 1):
        ta=time.time()
        print("\n=====Epoch {}".format(epoch), flush=True)
        ###BEGIN TRAIN
        loss_func = torch.nn.L1Loss()
        # train_mae = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device)
        model.train()
        loss_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out = model(batch_data)
            # loss = loss_func(out, batch_data.y.unsqueeze(1))
            # print("NOT CHANGE")
            loss = loss_func(out, batch_data.y)
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        train_mae = loss_accum / (step + 1)
        tb=time.time()
        save_time.append(tb-ta)
        save_loss.append(train_mae)
        #valid
        loss_accum_valid = 0
        for step_valid, batch_valid in enumerate(tqdm(valid_loader)):
            batch_valid = batch_valid.to(device)
            out_valid = model(batch_valid)
            loss_valid = loss_func(out_valid, batch_valid.y)
            loss_accum_valid += loss_valid.detach().cpu().item()
        valid_mae = loss_accum_valid / (step_valid + 1)
        save_loss_valid.append(valid_mae)
        print()
        print("MAE_TRAIN:", train_mae, "MAE_VALID", valid_mae)
        if epoch%n_ep_save==0:
            xas_pre_test = np.zeros([n_valid, npt])
            xas_true_test = np.zeros([n_valid, npt])
            i = 0
            loss_accum_valid = 0
            for step_valid, batch_valid in enumerate( tqdm(valid_loader_save) ):
                batch_valid = batch_valid.to(device)
                out_valid = model(batch_valid)
                loss_valid = loss_func(out_valid, batch_valid.y)
                loss_accum_valid += loss_valid.detach().cpu().item()

                xas_pre_test[i] = out_valid.cpu().detach().numpy()
                xas_true_test[i] = batch_valid.y.cpu().detach().numpy()
                # np.savetxt("xas_pre_test.txt", xas_pre_test)
                # np.savetxt("xas_true_test.txt", xas_true_test)
                i=i+1
            np.savetxt("xas_pre_test.txt", xas_pre_test)
            np.savetxt("xas_true_test.txt", xas_true_test)
            valid_mae = loss_accum_valid / (step_valid + 1)
            save_loss_valid.append(valid_mae)
            print()
            print("MAE_TRAIN:", train_mae, "SAVED_MAE_VALID", valid_mae)
            #site0
            # xas_pre_test = np.zeros([n_valid_site, npt])
            # xas_true_test = np.zeros([n_valid_site, npt])
            # i = 0
            # loss_accum_valid = 0
            # for step_valid, batch_valid in enumerate( tqdm(loader0) ):
            #     batch_valid = batch_valid.to(device)
            #     out_valid = model(batch_valid)
            #     loss_valid = loss_func(out_valid, batch_valid.y)
            #     loss_accum_valid += loss_valid.detach().cpu().item()
            #
            #     xas_pre_test[i] = out_valid.cpu().detach().numpy()
            #     xas_true_test[i] = batch_valid.y.cpu().detach().numpy()
            #     np.savetxt("xas_pre_site0.txt", xas_pre_test)
            #     np.savetxt("xas_true_site0.txt", xas_true_test)
            #     i=i+1
            #
            # valid_mae = loss_accum_valid / (step_valid + 1)
            # save_loss_valid.append(valid_mae)
            # print()
            # print("SITE0 MAE_TRAIN:", train_mae, "SAVED_MAE_VALID", valid_mae)

            # #site1
            # xas_pre_test = np.zeros([n_valid_site, npt])
            # xas_true_test = np.zeros([n_valid_site, npt])
            # i = 0
            # loss_accum_valid = 0
            # for step_valid, batch_valid in enumerate( tqdm(loader1) ):
            #     batch_valid = batch_valid.to(device)
            #     out_valid = model(batch_valid)
            #     loss_valid = loss_func(out_valid, batch_valid.y)
            #     loss_accum_valid += loss_valid.detach().cpu().item()
            #
            #     xas_pre_test[i] = out_valid.cpu().detach().numpy()
            #     xas_true_test[i] = batch_valid.y.cpu().detach().numpy()
            #     np.savetxt("xas_pre_site1.txt", xas_pre_test)
            #     np.savetxt("xas_true_site1.txt", xas_true_test)
            #     i=i+1
            #
            # valid_mae = loss_accum_valid / (step_valid + 1)
            # save_loss_valid.append(valid_mae)
            # print()
            # print("SITE1 MAE_TRAIN:", train_mae, "SAVED_MAE_VALID", valid_mae)
            #
            # #site2
            # xas_pre_test = np.zeros([n_valid_site, npt])
            # xas_true_test = np.zeros([n_valid_site, npt])
            # i = 0
            # loss_accum_valid = 0
            # for step_valid, batch_valid in enumerate( tqdm(loader2) ):
            #     batch_valid = batch_valid.to(device)
            #     out_valid = model(batch_valid)
            #     loss_valid = loss_func(out_valid, batch_valid.y)
            #     loss_accum_valid += loss_valid.detach().cpu().item()
            #
            #     xas_pre_test[i] = out_valid.cpu().detach().numpy()
            #     xas_true_test[i] = batch_valid.y.cpu().detach().numpy()
            #     np.savetxt("xas_pre_site2.txt", xas_pre_test)
            #     np.savetxt("xas_true_site2.txt", xas_true_test)
            #     i=i+1
            #
            # valid_mae = loss_accum_valid / (step_valid + 1)
            # save_loss_valid.append(valid_mae)
            # print()
            # print("SITE2 MAE_TRAIN:", train_mae, "SAVED_MAE_VALID", valid_mae)
            #
            # #site3
            # xas_pre_test = np.zeros([n_valid_site, npt])
            # xas_true_test = np.zeros([n_valid_site, npt])
            # i = 0
            # loss_accum_valid = 0
            # for step_valid, batch_valid in enumerate( tqdm(loader3) ):
            #     batch_valid = batch_valid.to(device)
            #     out_valid = model(batch_valid)
            #     loss_valid = loss_func(out_valid, batch_valid.y)
            #     loss_accum_valid += loss_valid.detach().cpu().item()
            #
            #     xas_pre_test[i] = out_valid.cpu().detach().numpy()
            #     xas_true_test[i] = batch_valid.y.cpu().detach().numpy()
            #     np.savetxt("xas_pre_site3.txt", xas_pre_test)
            #     np.savetxt("xas_true_site3.txt", xas_true_test)
            #     i=i+1

            # valid_mae = loss_accum_valid / (step_valid + 1)
            # save_loss_valid.append(valid_mae)
            # print()
            # print("SITE3 MAE_TRAIN:", train_mae, "SAVED_MAE_VALID", valid_mae)

        ###END TRAIN

        #

        if epoch%5==0:
            torch.save(model.state_dict(), 'tmp.pt')
            print("Model saved")
        # scheduler.step()

    np.savetxt("dig_loss.txt",np.array(save_loss))
    np.savetxt("dig_loss_valid.txt",np.array(save_loss_valid))
    save_time=np.array(save_time)
    print("TIME AVERAGE:",save_time.mean())

    res=save_loss_valid[-10:]
    res=np.array(res)
    res=res.mean()





    # aa=np.array([res,num_layers,hidden_channels,middle_channels,num_output_layers,num_radial,num_spherical])
    with open("hyper"+str(irun)+".log", "a+") as flog:
        print(res,num_layers,hidden_channels,middle_channels,num_output_layers,num_radial,num_spherical,file=flog)

    return res



def f2(params):
    num_layers=params['num_layers']
    hidden_channels=params['hidden_channels']
    print(num_layers,hidden_channels)
    # middle_channels=params['middle_channels']
    # num_output_layers=params['num_output_layers']
    # num_radial=params['num_radial']
    # num_spherical=params['num_spherical']
    return None


def run(irun):
    fname = "./name" + str(irun) + ".txt"
    parmat=np.loadtxt(fname).astype(int)
    if parmat.ndim==1:
        tmp=parmat
        ipar = dict(
            {'num_layers': tmp[0], 'hidden_channels': int(tmp[1]), 'middle_channels': tmp[2]
                , 'num_output_layers': tmp[3], 'num_radial': tmp[4], 'num_spherical': tmp[5]}
        )
        print(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
        f(ipar, irun)
    else:
        for i in range(parmat.shape[0]):
            tmp=parmat[i]
            ipar = dict(
            {'num_layers': tmp[0], 'hidden_channels': int(tmp[1]), 'middle_channels': tmp[2]
                , 'num_output_layers': tmp[3], 'num_radial': tmp[4], 'num_spherical': tmp[5]}
            )
            print(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5])
            f(ipar,irun)
    return None



# tmp=np.array([3,64,256,2,6,3])
# ipar = dict(
#     {'num_layers': tmp[0], 'hidden_channels': int(tmp[1]), 'middle_channels': tmp[2]
#         , 'num_output_layers': tmp[3], 'num_radial': tmp[4], 'num_spherical': tmp[5]}
# )
# print(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
# f(ipar,1)


if __name__ == '__main__':
    import sys
    ta=time.time()
    run( int(sys.argv[1]) )
    tb=time.time()
    with open("hyper_time"+str(sys.argv[1])+".log", "a+") as flog:
        print(tb-ta,file=flog)




# print("Grid search:")
# for num_layers in [3]:
#     for hidden_channels in [64,128,256]:
#         for middle_channels in [64,128,256]:
#             for num_output_layers in [2,3,4]:
#                 for num_radial in [3,6,12]:
#                     for num_spherical in [3,6]:
#
#                         ipar = dict(
#                             {'num_layers': num_layers, 'hidden_channels': hidden_channels, 'middle_channels': middle_channels
#                             ,'num_output_layers': num_output_layers, 'num_radial': num_radial, 'num_spherical': num_spherical})
#                         f(ipar)
