import numpy as np
from tqdm import tqdm #not import tqdm
import os.path as osp
import time
###
import torch
import torch_cluster
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,DirGNNConv,GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data,InMemoryDataset
from torch_geometric.loader import DataLoader

# import torch_geometric.transforms as T
# from torch.autograd import grad
# from torch_cluster import radius_graph
# from torch.optim.lr_scheduler import StepLR
# import torch.nn.functional as F
# from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log

Conv=GCNConv
class DirGNN(torch.nn.Module):
    def __init__(self,hidden_channels=128,in_channels=4, out_channels=120,n_layers=3, alpha=0.5,jk='lstm',dropout=0.5):
        super().__init__()
        self.dropout=dropout
        self.layers_in = Conv(in_channels, hidden_channels)
        self.layers_in = DirGNNConv(self.layers_in, alpha, root_weight=False)

        self.layers_out = Conv(hidden_channels, out_channels)
        self.layers_out = DirGNNConv(self.layers_out, alpha, root_weight=False)
        layers=[]
        for i in range(n_layers-2):
            tmp=Conv(hidden_channels , hidden_channels)
            tmp=DirGNNConv(tmp, alpha, root_weight=False)
            layers.append(tmp)
        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x, edge_index,batch_seg):
        
        x = F.relu(self.layers_in(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for ilayer in self.layers:           
            x = F.relu( ilayer(x, edge_index) )
            x = F.dropout(x, p=self.dropout, training=self.training)


        x = self.layers_out(x, edge_index)
        x=global_mean_pool(x, batch_seg)

        return x





        





########################################################################Setting
setting_prefix="Fe3O4_7113_17_21_39_site"
setting_n_spc=120#239
setting_batchsize=32
setting_num_epochs =1000#600
device = 'cuda' if torch.cuda.is_available() else 'cpu'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class dataset_train(InMemoryDataset):
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
        self.cutoff = 5.0
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix=setting_prefix
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_train_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_train_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_train_y.npy", allow_pickle=True)
        # r_list = np.load(prefix+"_train_r.npy",allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(int)
            z_double=torch.tensor(iz).double()
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            edge_index = torch_cluster.radius_graph(pos, r=self.cutoff)
            node_feature=torch.hstack( [z_double.reshape([-1,1]),pos] ).float()
            # r = torch.tensor(r_list[i])
            tmp = Data(x=node_feature, edge_index=edge_index,spectrum=y,pos=pos )#tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class dataset_test(InMemoryDataset):
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
        self.cutoff = 5.0
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix=setting_prefix
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_test_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_test_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_test_y.npy", allow_pickle=True)
        # r_list = np.load(prefix+"_train_r.npy",allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(int)
            z_double=torch.tensor(iz).double()
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            edge_index = torch_cluster.radius_graph(pos, r=self.cutoff)
            node_feature=torch.hstack( [z_double.reshape([-1,1]),pos] ).float()
            # r = torch.tensor(r_list[i])
            tmp = Data(x=node_feature, edge_index=edge_index,spectrum=y,pos=pos )#tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = dataset_train("train")
dataset_test = dataset_test("test")
print("Train length: ", len(dataset),"Val length: ", len(dataset_test)  )
print("Dataset")




train_loader = DataLoader(dataset, batch_size=setting_batchsize, shuffle=True)
val_loader = DataLoader(dataset_test, batch_size=setting_batchsize, shuffle=True)

save_loader = DataLoader(dataset_test, batch_size=setting_batchsize, shuffle=False)

print("Debug")

# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
#         # On the Pubmed dataset, use `heads` output heads in `conv2`.
#         self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads,dropout=0.6)
#         self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1,
#                              concat=False, dropout=0.6)

#     def forward(self, x, edge_index,batch_seg):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv3(x, edge_index)
#         x=global_mean_pool(x, batch_seg)

#         return x





global Nf
Nf=1
def f(params,irun):
# params = dict({'hidden_channels': 64, 'num_layers': 4, 'num_heads': 4, 'dropout': 0.8, 'lr': 0.0005})

    global Nf
    Nf=Nf+1
    print("Nf:",Nf)
    #3*3*3*2=54
    hidden_channels=params['hidden_channels'] #64 96 128 256
    num_layers=params['num_layers']#1 2 3 4 6 8
    lr=params['lr']#0.0005 0.001 


    model = DirGNN(n_layers=num_layers,hidden_channels=hidden_channels,in_channels=4,out_channels=setting_n_spc,).to(device)
    loss_fn=torch.nn.L1Loss()


    weight_decay=0;lr_decay_step_size=50;lr_decay_factor=0.5;
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)#optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    # lr =1e-3;milestones = np.arange(10, 100, 10).tolist()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.8)



    save_loss_valid=[]
    for epoch in range(setting_num_epochs):
        train_loss = 0
        num_molecules_train = 0
        val_loss = 0
        num_molecules_val = 0
        # model.train()###
        for step,batch in enumerate(tqdm(train_loader)):
            batch = batch.to(device)
            x, edge_index, batch_seg = batch.x, batch.edge_index, batch.batch
            optimizer.zero_grad()
            pred = model(x, edge_index, batch_seg)
            loss = loss_fn(pred.view(-1, 1),batch.spectrum.view(-1, 1))
            loss.backward()
            train_loss += loss
            num_molecules_train += batch.num_graphs
            optimizer.step()
        scheduler.step()   
        avg_train_loss = train_loss / (step+1) #avg_train_loss = train_loss / num_molecules_train
        
        
        with torch.no_grad():
            model.eval()
            for step,batch in enumerate(tqdm(val_loader)):
                batch = batch.to(device)
                x, edge_index, batch_seg = batch.x, batch.edge_index, batch.batch
                pred = model(x, edge_index, batch_seg)
                loss = loss_fn(pred.view(-1, 1), batch.spectrum.view(-1, 1))
                val_loss += loss
                num_molecules_val += batch.num_graphs
            avg_val_loss = val_loss / (step+1)# avg_val_loss = val_loss / num_molecules_val
            save_loss_valid.append(avg_val_loss.detach().cpu().item())

        print(f"epoch {epoch} | average train loss = {avg_train_loss:.5f}",f" and average validation loss = {avg_val_loss:.5f}")

        with open("loss.txt", "a") as f:
            f.write(f"{avg_train_loss:.4f}, {avg_val_loss:.4f}\n")
        if epoch%100==0:
            torch.save(model.state_dict(), 'tmp.pt')
            print("Model saved")

        yp=[]
        yt=[]
        with torch.no_grad():
            model.eval()
            if epoch%5==0: #and epoch>0:
                for batch in save_loader:
                    batch = batch.to(device)
                    x, edge_index, batch_seg = batch.x, batch.edge_index, batch.batch
                    pred = model(x, edge_index, batch_seg)
                    true = batch.spectrum.reshape(-1, setting_n_spc)
                    yp.append(pred.detach().cpu().numpy())
                    yt.append(true.detach().cpu().numpy()) 
                yp=np.vstack(yp)
                yt=np.vstack(yt)
                np.savetxt("yp.txt",yp)
                np.savetxt("yt.txt",yt)   
                print("Prediction saved",loss_fn(torch.Tensor(yp),torch.Tensor(yt))  )
    res=save_loss_valid[-10:]#save_loss_valid[-10:]
    res=np.array(res)
    res=res.mean()

    with open("hyper"+str(irun)+".log", "a+") as flog:
        print(res,hidden_channels,num_layers,lr,file=flog)#params = dict({'hidden_channels': 64, 'num_layers': 4, 'num_heads': 4, 'dropout': 0.8, 'lr': 0.0005})

    return res




def run(irun):
    fname = "./name" + str(irun) + ".txt"
    parmat=np.loadtxt(fname)
    if parmat.ndim==1:
        tmp=parmat
        ipar = dict(  dict({'hidden_channels': int(tmp[0]), 'num_layers': int(tmp[1]),  'lr': tmp[2]})  )
        print(tmp[0], tmp[1], tmp[2])
        f(ipar, irun)
    else:
        for i in range(parmat.shape[0]):
            tmp=parmat[i]
            ipar = dict(  dict({'hidden_channels': int(tmp[0]), 'num_layers': int(tmp[1]),  'lr': tmp[2]})  )
            print(tmp[0],tmp[1],tmp[2])
            f(ipar,irun)
    return None





if __name__ == '__main__':
    import sys
    ta=time.time()
    #run( 0 )
    #run( int(sys.argv[1]) )
    params = dict({'hidden_channels': 256, 'num_layers': 4,  'lr': 0.001})
    f(params,11)
    tb=time.time()
    with open("hyper_time"+str(sys.argv[1])+".log", "a+") as flog:
        print(tb-ta,file=flog)