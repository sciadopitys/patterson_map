import matplotlib
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

#from torchvision import transforms
import torch 
import torch.nn as nn
import torch.fft
import torch.cuda
#import torchvision.transforms as transforms
#import torchvision.datasets as dsets
import csv
import numpy as np
import sys
#from scipy import stats
#torch.manual_seed(0)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import math
import statistics
#from prettytable import PrettyTable
import dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #use GPU if possible
torch.backends.cudnn.benchmark = True

dataset.create_batches()

with open("training_indices_orth.txt") as myfile2: #select the first n_train examples as the training set, rest as validation set
    indices = myfile2.readlines()
indlist  = [x.rstrip() for x in indices]

with open("test_orth1.txt") as myfile: #select the first n_train examples as the training set, rest as validation set
    testlist = myfile.readlines()
testlist = [x.rstrip() for x in testlist]


class Dataset(torch.utils.data.Dataset):

  def __init__(self, pdbIDs):
        self.ids = pdbIDs

  def __len__(self):
        return len(self.ids)

  def __getitem__(self, index): #each example consists of a patterson map, electron density pair
        ID = self.ids[index]
        #print(ID)
        X = torch.load('patterson_pt_scaled_var2/' + ID + '_patterson.pt')
        X = torch.unsqueeze(X, 0)
        #X1 = torch.load('predictions13/' + ID + '.pt')
        #X1 = X1[0,0,:,:,:]
        X1 = torch.load('electron_density_pt_scaled_res/' + ID + '_a.pt')
        X1 = torch.unsqueeze(X1, 0)
        X2 = torch.load('electron_density_pt_scaled_res/' + ID + '_b.pt')
        X2 = torch.unsqueeze(X2, 0)
        #print(X.size(), X1.size(), X2.size())
        X3 = torch.cat((X, X1, X2), 0)
        #X3 = torch.cat((X, X1), 0)
        
        X3 = torch.unsqueeze(X3, 0)
        y = torch.load('electron_density_pt_scaled_var2/' + ID + '.pt')
        y = torch.unsqueeze(y, 0)
        #return X, y
        return X3, y  


dataset_val = Dataset(testlist)
n_test = float(len(dataset_val))


class Dataset1(torch.utils.data.Dataset):

    def __init__(self, indices): 
        self.indices = indices
        
    def __getitem__(self, index):
        X = torch.load('patterson_pt_scaled_var2/train_' + str(index) + '_patterson.pt')
        y = torch.load('electron_density_pt_scaled_var2/train_' + str(index) + '.pt')  
        
        return X, y
        
    def __len__(self):
        return len(self.indices) - 1
        
dataset_train = Dataset1(indlist)
n_train = len(indlist)


class SEBlock(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        
        self.fc1 = nn.Linear(c, c // r)
        self.fc2 = nn.Linear(c // r, c)
        
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        bs, c, _, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(bs, c, 1, 1, 1)
        
        return torch.mul(x, y.expand_as(x))


class ResSEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv7a = nn.Conv3d(in_channels=30, out_channels=30, kernel_size=7, padding=3, bias=False, padding_mode='circular')
        torch.nn.init.kaiming_normal_(self.conv7a.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm3d(30)
        
        self.conv7b = nn.Conv3d(in_channels=30, out_channels=30, kernel_size=7, padding=3, bias=False, padding_mode='circular')
        torch.nn.init.kaiming_normal_(self.conv7b.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm3d(30)
        
        self.se = SEBlock(30,2)
        
        self.relu = nn.ReLU()

    def forward(self,x):
        identity = x

        out = self.conv7a(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv7b(out)
        out = self.bn2(out)
        
        out = self.se(out)
        
        out += identity
        out = self.relu(out)

        return out

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv5a = nn.Conv3d(in_channels=3, out_channels=25, kernel_size=7, padding=3, bias=False, padding_mode='circular')
        torch.nn.init.kaiming_normal_(self.conv5a.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm3d(25)
        
        self.conv5b = nn.Conv3d(in_channels=25, out_channels=30, kernel_size=7, padding=3, bias = False, padding_mode='circular')
        torch.nn.init.kaiming_normal_(self.conv5b.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm3d(30)   
        
        self.pool=nn.MaxPool3d(kernel_size=2)
        
        self.conv7block = nn.ModuleList()
        for _ in range(8):
            self.conv7block.append(ResSEBlock())
            
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv5c = nn.Conv3d(in_channels=30, out_channels=25, kernel_size=5, padding=2, bias=False)
        torch.nn.init.kaiming_normal_(self.conv5c.weight, nonlinearity='relu')
        self.bn3 = nn.BatchNorm3d(25)
        
        self.output = nn.Conv3d(in_channels=25, out_channels=1, kernel_size=5, padding=2)
        torch.nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu')
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.squeeze(x,0)
        x = self.conv5a(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv5b(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        for layer in self.conv7block:
            x = layer(x)
            
        x = self.upsamp(x)
        x = self.conv5c(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.output(x)
        x = torch.tanh(x)
        return x


epsilon = 1e-8

def pearson_r_loss(output, target): #calculate pearson r coefficient for central region
    #x = output[:,:,14:26,14:26,14:26]
    #y = target[:,:,14:26,14:26,14:26]

    x = output
    y = target
    
    #x = output[:,:,22:38,22:38,22:38]
    #y = target[:,:,22:38,22:38,22:38]   
    
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return cost
    
def pearson_r_loss2(output, target): #calculate pearson r coefficient for central region
    #x = output[:,:,14:26,14:26,14:26]
    #y = target[:,:,14:26,14:26,14:26]
    x = output[:,0,:,:,:]
    y = torch.squeeze(target, 0)[:,0,:,:,:]
    
    #x = output[:,:,22:38,22:38,22:38]
    #y = target[:,:,22:38,22:38,22:38]   
    
    batch = x.shape[0]
    cost = 0.0
    
    for i in range(batch):
    
        curx = x[i,:,:,:]
        cury = y[i,:,:,:]
        
        vx = curx - torch.mean(curx)
        vy = cury - torch.mean(cury)

        cost += (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return (cost / batch)
    
def fft_loss(patterson, electron_density):
    patterson = patterson[0,0,0,:,:]
    electron_density = electron_density[0,0,:,:,:]
    f1 = torch.fft.fftn(electron_density)
    f2 = torch.fft.fftn(torch.roll(torch.flip(electron_density, [0, 1, 2]), shifts=(1, 1, 1), dims=(0, 1, 2)))
    f3 = torch.mul(f1,f2)
    #f3 = torch.mul(f1.real,f1.real)
    f4 = torch.fft.ifftn(f3)
    f4 = f4.real

    vx = f4 - torch.mean(f4)
    vy = patterson - torch.mean(patterson)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return cost

#specify batch size
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, shuffle = True, batch_size= 1, num_workers = 4, pin_memory = True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_val, shuffle = False, batch_size= 1, num_workers = 4, pin_memory = True)

model = CNN()
model.to(device)

#loading pretrained model
#checkpoint = torch.load('state_dipeptide_tanh1.pth')
#model.load_state_dict(checkpoint['model_state_dict'])

#specify loss function, learning rate schedule, number of epochs
criterion = nn.MSELoss()
#learning_rate = 4e-5
learning_rate = 1.2e-4
n_epochs = 1000
epoch = 0
accum = 4  #gradient accumulation; effective batch size is 13x
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=3e-2)
#lambda1 = lambda epoch: (0.86 ** epoch) if (epoch <= 9) else ((0.86 ** 9) * (0.999 ** (epoch - 9)))
lambda1 = lambda epoch: (0.873 ** epoch) if (epoch <= 10) else ((0.873 ** 10) * (0.9991 ** (epoch - 10)))
#lambda1 = lambda epoch: (0.858 ** epoch) if (epoch <= 9) else (((0.85 ** 9) * (0.999 ** (epoch - 18))) + ((0.999 ** (epoch - 14)) * 0.01)*math.sin(0.225*epoch-16))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#loss = checkpoint['loss']
#epoch = checkpoint['epoch']

def mse_wrapper_loss(output, target):

    y = torch.squeeze(target, 0)
    return criterion(output, y)

t1 = time.perf_counter()
clip = 1.0 #gradient clipping value
test1 = []
test2 = []
while epoch < n_epochs:
    model.train() 
    acc = 0.0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        yhat = model(x)                                                 #apply model to current example
        loss_1 = mse_wrapper_loss(yhat, y)                              #evaluate loss
        loss_2 = (1 - pearson_r_loss2(yhat, y))
        loss = (0.999 * loss_1) + (1e-3 * loss_2)
        #print(float(loss.item()))
        acc += float(loss.item())
        loss = loss / accum                                             #needed due to gradient accumulation
        loss.backward()                                                 #compute and accumulate gradients for model parameters
        if (i+1) % accum == 0:                                          
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    #gradient clipping
            optimizer.step()                                            #update model parameters only on accumulation epochs
            optimizer.zero_grad()                                       #clear (accumulated) gradients
            torch.cuda.empty_cache()
            
    model.eval() 
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    with torch.no_grad(): #calculate loss and pearson r for all validation set elements
        for x, y in test_loader: 
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss3 = criterion(yhat, y)
            loss4 = pearson_r_loss(yhat, y)
            loss5 = fft_loss(x, yhat)
            acc1 += float(loss3.item())
            acc2 += float(loss4.item())
            acc3 += float(loss5.item())
            torch.cuda.empty_cache()
    
    #store average value of metrics
    test1.append(acc1 / n_test)
    curacc = (acc2 / n_test)
    curacc2 = (acc3 / n_test)
    test2.append(curacc)
    print("%d %.10f %.6f %.6f %.10f" % (epoch, (acc / n_train), curacc, curacc2, scheduler.get_last_lr()[0]))
    #print(acc / n_train)
    #print(curacc)
    scheduler.step()
    
    
    if (epoch % 3 == 0):
        
        dataset.create_batches()
        
    if (epoch % 5 == 0):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'epoch': epoch + 1,
            }, 'state_dipeptide_tanh1.pth')
     
    #dataset.create_batches()
    epoch += 1
t2 = time.perf_counter()
print(t2-t1)


torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'epoch': epoch + 1,
            }, 'state_dipeptide_tanh1.pth')
"""

#evaluate final model on validation set
model.eval() 
losses1 = []    
losses2 = []  
count = 0
with torch.no_grad():
    for x, y in test_loader: 
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        loss1 = fft_loss(x, yhat)
        loss2 = pearson_r_loss(yhat, y)
        #print(float(loss1.item()))
        #if ((float(loss2.item())) > 0.9975) or ((float(loss2.item())) < 0.6):
        #    print(count)
        #    print(float(loss2.item()))
        losses1.append(float(loss1.item()))
        losses2.append(float(loss2.item()))
        if count < 5:    
            yhatc = yhat.cpu()
            print(float(loss1.item()))
            print(float(loss2.item()))
            torch.save(yhatc, 'predictions_res/' + testlist[count] + '.pt')
        torch.cuda.empty_cache()
        count += 1


hist, bin_edges = np.histogram(losses1)
print(hist)
print(bin_edges)

print(sum(losses1) / len(losses1))
print(statistics.pstdev(losses1))

hist, bin_edges = np.histogram(losses2)
print(hist)
print(bin_edges)

print(sum(losses2) / len(losses2))
print(statistics.pstdev(losses2))

#create plot of average validation set metrics over epochs
#print(*test1, sep = "\n")
#print(*test2, sep = "\n")
x = np.arange(1, n_epochs + 1)
plt.plot(x,test1)
plt.plot(x,test2)
plt.show()
"""