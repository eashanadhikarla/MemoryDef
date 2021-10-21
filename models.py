'''
Authors  : Eashan Adhikarla, Dan Luo, Dr. Brian D. Davison
Paper    : Memory Defense: More Robust Classification 
           via a Memory-Masking Autoencoder
        
Permission is granted to anyone to use this repository for any purpose,
including commercial applications, and to alter it and redistribute it
freely.

'''

from __future__ import absolute_import, print_function

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import os, pickle, math, logging, random

device = 'cpu'

def masking(s, label, mem_dim, device):
    memoryPerClass = mem_dim//10
    batch_size = len(label)
    
    mask1      = torch.zeros(batch_size, mem_dim)
    mask2      = torch.ones(batch_size, mem_dim)
    ones       = torch.ones(memoryPerClass)
    zeros      = torch.zeros(memoryPerClass)
    
    for i in range(batch_size):
        lab    = torch.arange(memoryPerClass*label[i], memoryPerClass*(label[i]+1), dtype=torch.long)
        if lab.nelement()==0:
            print("Label tensor empty in the memory module.")
        else:
            mask1[i,lab] = ones
            mask2[i,lab] = zeros     
    return mask1.to(device), mask2.to(device)

class MemoryDefense(nn.Module):
    def __init__(self, MEM_DIM, device):
        super(MemoryDefense, self).__init__()

        self.device = device
        self.num_classes = 10
        
        self.image_height = 28
        self.image_width = 28
        self.image_channel_size = 1
        
        self.num_memories = MEM_DIM
        self.addressing = 'sparse'
        
        self.conv_channel_size = 4
        self.feature_size = self.conv_channel_size * 16 * 4 * 4
        self.drop_rate = 0.5
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        
        # Encoder
        self.encoder = Encoder()
        
        # Memory
        init_mem = torch.zeros(self.num_memories, self.feature_size)
        '''
        Deep NN models have difficulties in converging when the weights are
        initialized using Normal Distribution with fixed standard deviation. 
        This is because the variance of weights is not taken care of, which 
        leads to very large or small activation values, resulting in exploding
        or vanishing gradient problem during backpropagation.
        '''
        nn.init.kaiming_uniform_(init_mem)
        self.memory = nn.Parameter(init_mem)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        '''
        To sparsify the weight matrix, so that it does not become capable 
        of reconstructing anomaly and adversarial equally well.
        '''
        self.hardshrink = nn.Hardshrink(lambd=1e-12) # 1e-12
        
        # Decoder
        self.decoder = Decoder()
        
        if self.addressing == 'sparse':
            self.threshold = 1 / self.memory.size(0)
            self.epsilon = 1e-12
        
        self.classifier = Classifier(image_channel_size=self.image_channel_size, num_classes=self.num_classes, 
                                     drop_rate=self.drop_rate)
        
    def forward(self, x, labels, device='cuda'):
        batch, channel, height, width = x.size()
        
        # Encoder
        x = self.encoder(x)
        batch, _, _, _ = x.size()
        z = x.view(batch, -1)
        
        # Memory
        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1) 
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)
        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)
        
        # Masking using one hot encoding scheme over memory slots.
        m1, m2 = masking(x.data.shape, labels, self.num_memories, device) # Generating Mask
        masked_mem_weight = mem_weight*m1     # Masking target class
        masked_mem_weight_hat = mem_weight*m2 # Masking non-target class
        
        if self.addressing == 'soft':
            z_hat_target = torch.mm(masked_mem_weight, self.memory) # Matrix Multiplication:  Att_W x Mem
            z_hat_non_target = torch.mm(masked_mem_weight_hat, self.memory) # Matrix Multiplication:  Att_W x Mem
            
        elif self.addressing == 'sparse':
            # Unmask Weight Target Class
            masked_mem_weight = self.hardshrink(masked_mem_weight)
            masked_mem_weight = masked_mem_weight / masked_mem_weight.norm(p=1, dim=1).unsqueeze(1).expand(batch, self.num_memories)
            z_hat_target = torch.mm(masked_mem_weight, self.memory)
            # Mask Weight Non-target Class
            masked_mem_weight_hat = self.hardshrink(masked_mem_weight_hat)
            masked_mem_weight_hat = masked_mem_weight_hat / masked_mem_weight_hat.norm(p=1, dim=1).unsqueeze(1).expand(batch, self.num_memories)
            z_hat_non_target = torch.mm(masked_mem_weight_hat, self.memory)
        
        # Decoder
        rec_x     = self.decoder(z_hat_target)
        rec_x_hat = self.decoder(z_hat_non_target)
        
        # Target Classifier
        logit_x = self.classifier(rec_x)
        
        return dict(logit_x=logit_x, encoded = z_hat_target,
                    rec_x=rec_x, rec_x_hat=rec_x_hat,
                    mem_weight=masked_mem_weight, mem_weight_hat=masked_mem_weight_hat)
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        self.en = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,1), stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.25, True),

            nn.Conv2d(16, 32, kernel_size=(3,3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.25, True),

            nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.en(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_channel_size = 16
        self.dec = nn.Sequential(            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(0,0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.25, True),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.25, True),

            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=(1,1))
            )
        
    def forward(self, x):
        x = x.view(-1, self.conv_channel_size * 4 , 4 , 4)
        x = self.dec(x)
        return x

    
class Classifier(nn.Module):
    def __init__(self, image_channel_size, num_classes, drop_rate):
        super(Classifier, self).__init__()
        self.image_channel_size = image_channel_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(in_channels=self.image_channel_size, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x = self.relu(self.maxpool2d(self.conv1(x)))
        x = self.relu(self.maxpool2d(self.conv2(x)))
        # print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x