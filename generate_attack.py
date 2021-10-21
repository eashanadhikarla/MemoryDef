#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, print_function

from models import MemoryDefense
from resnet import resnet18, resnet50
from wrn import wrn
from util import *

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import os, pickle, math, logging, random, time as t

# Advertorch Library for Adversarial Attack
from py_attacks.one_step_gradient import GradientSignAttack
from py_attacks.iterative_projected_gradient import LinfBasicIterativeAttack
from py_attacks.iterative_projected_gradient import LinfPGDAttack
from py_attacks.iterative_projected_gradient import L2PGDAttack
from py_attacks.carlini_wagner import CarliniWagnerL2Attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "~/fmnist"

epsilons     = [0.03, 0.09]
epsilon      = epsilons[0]
CONF         = 0
key          = 'pgd_linf'
MEM_DIM      = 500
NC           = 10

BATCHSIZE = 100

classifier_A  = resnet50(device=device)
classifier_A.load_state_dict(torch.load(str(root_dir)+"/checkpoint/resnet50.pt"))
classifier_A = nn.DataParallel(classifier_A) if torch.cuda.device_count() > 1 else classifier_A
classifier_A.to(device)
classifier_A.eval()

autoencoder = MemoryDefense(MEM_DIM, device)
autoencoder.load_state_dict(torch.load(str(root_dir)+"/checkpoint/memory_defense.pt"))
autoencoder = nn.DataParallel(autoencoder) if torch.cuda.device_count() > 1 else autoencoder
autoencoder.to(device)
autoencoder.eval()

cudnn.benchmark = True

# Attack dictionary: {FGSM, BIM, PGD, CW}
attack_dict = {
    "fgsm"    : GradientSignAttack       (autoencoder, loss_fn=None, eps=epsilon,
                                          clip_min=0.0, clip_max=1.0, targeted=False),
    
    "bim"     : LinfBasicIterativeAttack (autoencoder, loss_fn=None, eps=epsilon,
                                          nb_iter=10, eps_iter=0.05, clip_min=0.0,
                                          clip_max=1.0, targeted=False),
    
    "pgd_linf": LinfPGDAttack            (autoencoder, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                          eps=epsilon, nb_iter=40, eps_iter=0.0075, rand_init=True, 
                                          clip_min=0.0, clip_max=1.0, targeted=False),
    
    "cw_l2"   : CarliniWagnerL2Attack    (autoencoder, num_classes=10, confidence=CONF,
                                          targeted=False, learning_rate=0.01, binary_search_steps=9,
                                          max_iterations=1000, abort_early=True, initial_const=0.001,
                                          clip_min=0.0, clip_max=1.0, loss_fn=None),
    }

attack = attack_dict[key]
print(f"=============================================\nAttack  : {key}\nEpsilon : {epsilon+0:03}")
print(f"=============================================\nNunber of GPU's in use: {torch.cuda.device_count()}\n=============================================\n")

transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

traindata   = torchvision.datasets.FashionMNIST(
    root=str(root_dir)+"/data",
    train=True, 
    transform=transform_train
    )
trainloader = DataLoader(
    traindata,
    batch_size=BATCHSIZE,
    shuffle=True,
    num_workers=8
    )
testdata    = torchvision.datasets.FashionMNIST(
    root=str(root_dir)+"/data",
    train=False,
    transform=transform_test
    )
testloader  = DataLoader(
    testdata,
    batch_size=BATCHSIZE,
    shuffle=True,
    num_workers=8
    )

# =============================================================================
# Preparing Training Data
# =============================================================================
def main(epsilon, opt, attack):
    adv_images, adv_labels = [], []
    if opt=='train':
        dataloader = trainloader
        if key=='cw_l2':
            filename_adv_images = str(root_dir)+"/data/train/"+str(training)+"/"+str(training)+"_images_"+str(key)+"_"+str(conf)
            filename_adv_labels = str(root_dir)+"/data/train/"+str(training)+"/"+str(training)+"_labels_"+str(key)+"_"+str(conf)
        else:
            filename_adv_images = str(root_dir)+"/data/train/"+str(training)+"/"+str(training)+"_images_"+str(key)+"_"+str(epsilon)
            filename_adv_labels = str(root_dir)+"/data/train/"+str(training)+"/"+str(training)+"_labels_"+str(key)+"_"+str(epsilon)
    elif opt=='test':
        dataloader = testloader
        if key=='cw_l2':
            filename_adv_images = str(root_dir)+"/data/test/"+str(training)+"/"+str(training)+"_images_"+str(key)+"_"+str(conf)
            filename_adv_labels = str(root_dir)+"/data/test/"+str(training)+"/"+str(training)+"_labels_"+str(key)+"_"+str(conf)
        else:
            filename_adv_images = str(root_dir)+"/data/test/"+str(training)+"/"+str(training)+"_images_"+str(key)+"_"+str(epsilon)
            filename_adv_labels = str(root_dir)+"/data/test/"+str(training)+"/"+str(training)+"_labels_"+str(key)+"_"+str(epsilon)
            
    bar = tqdm(total=len(dataloader.dataset), desc='Loop', position=0)
    for batch_idx, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)
        # ============= Benign =============
        for im,lab in zip(image,label):
            adv_images.append( im.to("cpu") )
            adv_labels.append( lab.to("cpu") )
        # =========== Adversarial ===========
        perturbed_image = attack.perturb(image, label)
        
        ## ========================================================
        ## Comment this when training Classifier-A
        ## ========================================================
        #---------- Classifier A ----------
        predict_A = classifier_A(perturbed_image)
        pred_label = torch.max(predict_A.data, 1)[1]
        #---------- Stacked Model {Autoencoder & Classifier B} ----------
        output_dict = autoencoder(perturbed_image, pred_label, device)
        pred = output_dict['logit_x']
        predict = torch.max(pred.data, 1)[1]
        predict = torch.max(predict_A.data, 1)[1]
        
        for im2, lab2, p in zip(perturbed_image, label, predict):
            # if p!=lab2:
            im2 = im2.squeeze_(1)
            adv_images.append( im2.to("cpu") )
            adv_labels.append( lab2.to("cpu") )
            bar.update(1)
        ## ========================================================

        ## ========================================================
        ## Uncomment this when training Classifier-A
        ## ========================================================
        # for im2, lab2 in zip(perturbed_image, label):
        #     im2 = im2.squeeze_(1)
        #     adv_images.append( im2.to("cpu") )
        #     adv_labels.append( lab2.to("cpu") )
        #     bar.update(1)
            
    torch.save(adv_images, filename_adv_images)
    torch.save(adv_labels, filename_adv_labels)
    print('==============================================')
    print(f'Data Prepared, Total examples - {len(adv_images)}')
    print('==============================================')

if __name__ == "__main__":
    start = t.time()
    print('Starting data generation...\n')
    
    opt = 'train'
    main(epsilon, opt, attack)
    print("\n")
    opt = 'test'
    main(epsilon, opt, attack)

    end = (t.time()-start)
    end = float(end/60)
    print('\nExecution time: ', end)
    print('==============================================')


def preprocessing_(key, epsilon, training='stack_adversarial'): # training='classifier'

    for opt in ['train', 'test']:
        file_adv_im=str(root_dir)+"/data/"+str(opt)+"/"+str(training)+"/"+str(training)+"_images_"+str(key)+"_"+str(epsilon)
        file_adv_lab=str(root_dir)+"/data/"+str(opt)+"/"+str(training)+"/"+str(training)+"_labels_"+str(key)+"_"+str(epsilon)
        
        img = torch.load(file_adv_im)
        lab = torch.load(file_adv_lab)
        
        xs = torch.stack(img)
        xs = xs.squeeze(1)
        ys = torch.Tensor(lab)
        dataset = TensorDataset(xs, ys)
        
        if opt=='train':
            train_loader = dataset
        elif opt=='test':
            test_loader  = dataset
    print('Data loaded successfully.\n')
    return train_loader, test_loader

epsilons  = [0.03, 0.09]
epsilon   = epsilons[0]
key       = 'pgd_linf'

traindata, testdata = preprocessing_(key, epsilon)
if len(traindata)!=0 and len(testdata)!=0:
    print("Data Processed!")
