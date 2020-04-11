import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import glob
from torch.utils import data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import random
import statistics

path = 'C:/Users/Nolan/Desktop/letters/npy'
current = os.getcwd()
os.chdir(path)

data_npy={}
for file in glob.glob("*.npy"):
    data_npy[file[0]]=np.load(file)
os.chdir(current)


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.labels = list(range(len(data.keys())))
        key_labels = data.keys()
        info=[]
        for things in key_labels:
            info.append(data[things].astype(np.float32))
        self.data=info
        info=[]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)
        #return 100

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        num=random.uniform(0,1)
        array = self.data[index]
        ID = self.labels[index]
        if num<0.5:
            out= np.flipud(self.data[index]).copy()
        else:
            out = array
            
        num=random.uniform(0,1)
        
        if num<0.5:
            out2= np.fliplr(out).copy()
        else:
            out2 = out

        
        return ID, out2

class ConvNetwork(nn.Module):
  def __init__(self):
    super(ConvNetwork,self).__init__()
    Conv3d=nn.Conv3d
    self.c1=Conv3d(1,2,(2,2,3),stride=(2,2,1),padding=(0,0,1))#go from 100,100,5 to 50,50,5
    self.c2=Conv3d(2,5,(2,2,3),stride=(2,2,1),padding=(0,0,1))#go from 50,50,5 to 25,25,5
    self.c3=Conv3d(5,10,(3,3,2),stride=(2,2,1),padding=(0,0,0))#go from 25,25,5 to 12,12,4
    self.c4=Conv3d(10,25,(2,2,2),stride=(2,2,2),padding=(0,0,0))#go from 12,12,4 to 6,6,2
    self.c5=Conv3d(25,25,(6,6,2),stride=(1,1,2),padding=(0,0,0))#go from 6,6,2 to 1,1,1
    self.activation = nn.ReLU()
    
    
  def forward(self,x):
    x1=self.activation(self.c1(x))
    x2=self.activation(self.c2(x1))
    x3=self.activation(self.c3(x2))
    x4=self.activation(self.c4(x3))
    x5=(self.c5(x4))
    

    return x5
    
epochs=200
#define the objective function
objective = torch.nn.CrossEntropyLoss()
model = ConvNetwork()

optimizer = optim.Adam(model.parameters(),lr=1e-4)


params={'batch_size':5,'shuffle':True}
training_set=Dataset(data_npy)
training_generator=data.DataLoader(training_set,**params)
loop = tqdm(total=epochs, position = 0)
loss_record=[]
loss_temp=[]
for i in range(epochs):
    for things in training_generator:
        array = things[1].unsqueeze(1)
        y = things[0]
        optimizer.zero_grad()
        yhat = model(array).squeeze(-1).squeeze(-1).squeeze(-1)
        #print(yhat.size())
        loss = objective(yhat, y)
        loss.backward()
        optimizer.step()
        loss_temp.append(loss.item())
    loss_record.append(statistics.mean(loss_temp))
    print(statistics.mean(loss_temp))
    loop.update(1)
    
plt.plot(loss_record)
