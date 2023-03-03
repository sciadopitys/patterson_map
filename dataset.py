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


import random
def shuffle_slice(a, start, stop):
    i = start
    while (i < (stop-1)):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1


def create_batches():
    
    with open("training_orth.txt") as myfile1: #select the first n_train examples as the training set, rest as validation set
        trainlist = myfile1.readlines()
    trainlist  = [x.rstrip() for x in trainlist]
    

    with open("size_indices_orth.txt") as myfile: #select the first n_train examples as the training set, rest as validation set
        sindices = myfile.readlines()
    sindices  = [x.rstrip() for x in sindices]
    
    for i in range(len(sindices) - 1):
        start = int(sindices[i])
        end = int(sindices[i+1])
        shuffle_slice(trainlist, start, end)
        

    with open("training_indices_orth.txt") as myfile2: #select the first n_train examples as the training set, rest as validation set
        indices = myfile2.readlines()
    indices  = [x.rstrip() for x in indices]

    for i in range((len(indices) - 1)):
        start = int(indices[i])
        end = int(indices[i+1])
        xlist = []
        ylist = []
        for j in range(start, end):
            new_x = torch.load('patterson_pt_scaled_var2/' + trainlist[j] + '_patterson.pt')
            #new_x1 = torch.load('predictions13/' + trainlist[j] + '.pt')
            #new_x1 = new_x1[0,0,:,:,:]
            new_x1 = torch.load('electron_density_pt_scaled_res/' + trainlist[j] + '_a.pt')
            new_x2 = torch.load('electron_density_pt_scaled_res/' + trainlist[j] + '_b.pt')
            new_x = torch.unsqueeze(new_x, 0)
            new_x1 = torch.unsqueeze(new_x1, 0)
            new_x2 = torch.unsqueeze(new_x2, 0)
            #print(trainlist[j])
            #print(new_x.size(), new_x1.size(), new_x2.size())
            new_xcomb = torch.cat((new_x, new_x1, new_x2), 0)
            #new_xcomb = torch.cat((new_x, new_x1), 0)
            xlist.append(new_x)
            xlist.append(new_xcomb)
            new_y = torch.load('electron_density_pt_scaled_var2/' + trainlist[j] + '.pt')
            new_y = torch.unsqueeze(new_y, 0)
            ylist.append(new_y)
        
        data_x = torch.stack(xlist)
        data_y = torch.stack(ylist)
        torch.save(data_x, 'patterson_pt_scaled_var2/train_' + str(i) + '_patterson.pt')
        torch.save(data_y, 'electron_density_pt_scaled_var2/train_' + str(i) + '.pt')
    