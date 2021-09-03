import matplotlib
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import csv
import numpy as np
from scipy import stats
#torch.manual_seed(0)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

n_train = 28470
#n_train = 0
with open("full_list_shuf.txt") as myfile:
    trainlist = [next(myfile) for x in range(n_train)]
    testlist = myfile.readlines()
trainlist = [x.rstrip() for x in trainlist]
testlist = [x.rstrip() for x in testlist]

class Dataset(torch.utils.data.Dataset):

  def __init__(self, pdbIDs):
        self.ids = pdbIDs

  def __len__(self):
        return len(self.ids)

  def __getitem__(self, index):
        ID = self.ids[index]
        X = torch.load('diala_patterson_pt_scaled/' + ID + '_patterson.pt')
        X = torch.unsqueeze(X, 0)
        y = torch.load('diala_electron_density_pt_scaled/' + ID + '.pt')
        y = torch.unsqueeze(y, 0)
        return X, y


dataset_train = Dataset(trainlist)
dataset_val = Dataset(testlist)


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv7a = nn.Conv3d(in_channels=23, out_channels=23, kernel_size=7, padding=3)
        torch.nn.init.kaiming_normal_(self.conv7a.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm3d(23)
        
        self.conv7b = nn.Conv3d(in_channels=23, out_channels=23, kernel_size=7, padding=3)
        torch.nn.init.kaiming_normal_(self.conv7b.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm3d(23)
        
        self.relu = nn.ReLU()

    def forward(self,x):
        identity = x

        out = self.conv7a(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv7b(out)
        
        out += identity
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Conv7(nn.Module):
    def __init__(self):
        super(Conv7, self).__init__()
        self.conv7 = nn.Conv3d(in_channels=23, out_channels=23, kernel_size=7, padding=3)
        torch.nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='relu')
        self.bn = nn.BatchNorm3d(20)
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv7(x)
        #x = self.bn(x)
        x = self.relu(x)
        return x

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv5a = nn.Conv3d(in_channels=1, out_channels=23, kernel_size=5, padding=2)
        torch.nn.init.kaiming_normal_(self.conv5a.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm3d(23)
        
        self.conv5b = nn.Conv3d(in_channels=23, out_channels=23, kernel_size=5, padding=2)
        torch.nn.init.kaiming_normal_(self.conv5b.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm3d(23)
        
        self.pool=nn.MaxPool3d(kernel_size=2)
        
        self.conv7block = nn.ModuleList()
        for _ in range(5):
            self.conv7block.append(ResBlock())
            
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv5c = nn.Conv3d(in_channels=23, out_channels=23, kernel_size=5, padding=2)
        torch.nn.init.kaiming_normal_(self.conv5c.weight, nonlinearity='relu')
        self.bn3 = nn.BatchNorm3d(23)
        
        self.output = nn.Conv3d(in_channels=23, out_channels=1, kernel_size=5, padding=2)
        torch.nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu')
        
        self.relu = nn.ReLU()

    def forward(self, x):
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

def pearson_r_loss(output, target):
    x = output[:,:,14:26,14:26,14:26]
    y = target[:,:,14:26,14:26,14:26]
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.square(vx)) + epsilon) * torch.sqrt(torch.sum(torch.square(vy)) + epsilon)))
    return cost

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, shuffle = True, batch_size= 146, num_workers = 4, pin_memory = True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_val, shuffle = False, batch_size= 1, num_workers = 4, pin_memory = True)

model = CNN()
model.to(device)

#checkpoint = torch.load('model/state.pth')
#model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.MSELoss()
learning_rate = 5e-5
n_epochs = 1000
accum = 3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
lambda1 = lambda epoch: (0.6975 ** epoch) if (epoch <= 8) else (0.056021366 * (0.9991 ** (epoch - 8)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#loss = checkpoint['loss']

t1 = time.perf_counter()
clip = 1.0
test1 = []
test2 = []
for epoch in range(n_epochs):
    model.train() 
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        loss = criterion(yhat, y)
        #loss = pearson_r_loss(yhat, y)
        print(float(loss.item()))
        loss = loss / accum        
        loss.backward()
        if (i+1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
    model.eval() 
    acc1 = 0.0
    acc2 = 0.0
    with torch.no_grad():
        for x, y in test_loader: 
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss1 = criterion(yhat, y)
            loss2 = pearson_r_loss(yhat, y)
            acc1 += float(loss1.item())
            acc2 += float(loss2.item())
            torch.cuda.empty_cache()
    
    test1.append(acc1 / 3147.0)
    test2.append(acc2 / 3147.0)
    scheduler.step()

t2 = time.perf_counter()
print(t2-t1)

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            }, 'model1/state.pth')

model.eval() 
losses1 = []    
losses2 = []  
count = 0
with torch.no_grad():
    for x, y in test_loader: 
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        loss1 = criterion(yhat, y)
        loss2 = pearson_r_loss(yhat, y)
        #print(float(loss.item()))
        #if ((float(loss.item())) < 0.7):
        #    print(count)
        #    print(float(loss.item()))
        losses1.append(float(loss1.item()))
        losses2.append(float(loss2.item()))
        yhatc = yhat.cpu()
        torch.save(yhatc, 'predictions1/' + (str(count)) + '.pt')
        torch.cuda.empty_cache()
        count += 1
"""
with open('document.csv','a') as fd:
    writer = csv.writer(fd)
    for i in range(40):
        writer.writerow([str((yhat[0][0][i][x][x]).item()) for x in range(40)])
              
with open('document.csv','a') as fd:
    writer = csv.writer(fd)
    for i in range(40):
        for j in range(40):
            for k in range(10):
                writer.writerow([str((yhat[0][0][i][j][4*k]).item()), str((yhat[0][0][i][j][4*k + 1]).item()), str((yhat[0][0][i][j][4*k + 2]).item()), str((yhat[0][0][i][j][4*k + 3]).item())])
"""
hist, bin_edges = np.histogram(losses1)
print(hist)
print(bin_edges)

hist, bin_edges = np.histogram(losses2)
print(hist)
print(bin_edges)

print(*test1, sep = "\n")
print(*test2, sep = "\n")
x = np.arange(1, n_epochs + 1)
plt.plot(x,test1)
plt.plot(x,test2)
plt.show()