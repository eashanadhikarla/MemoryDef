'''
Authors  : Eashan Adhikarla, Dan Luo, Dr. Brian D. Davison
Paper    : Memory Defense: More Robust Classification 
           via a Memory-Masking Autoencoder
        
Permission is granted to anyone to use this repository for any purpose,
including commercial applications, and to alter it and redistribute it
freely.

'''

from __future__ import absolute_import, print_function

from models import MemoryDefense
from resnet import resnet50
from util import *

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import os, pickle, math, logging, random, time

import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

from advertorch_attacks.advertorch.attacks.one_step_gradient import GradientSignAttack
from advertorch_attacks.advertorch.attacks.iterative_projected_gradient import LinfBasicIterativeAttack, LinfPGDAttack, L2PGDAttack
from advertorch_attacks.advertorch.attacks.carlini_wagner import CarliniWagnerL2Attack

parser = argparse.ArgumentParser(description='Memory Defense')
parser.add_argument('-m', '--trained_model', default='./checkpoint/memory_defense_adamW.pt',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cA', default='resnet50', 
                    help='Classifier-A: resnet50')
parser.add_argument('--nc', default=10, type=int, 
                    help='Number of Classes')
parser.add_argument('--dim', default=500, type=int, 
                    help='Memory Weight for reconstruction')
parser.add_argument('--batch', default=1, type=int, 
                    help='Batch-size')

parser.add_argument('--eps', default=0.03, type=float, 
                    help='EPSILON: [0.03, 0.09]')
parser.add_argument('--attack', default='bim', 
                    help='Advertorch Attacks: {fgsm, bim, pgd, cw}')
parser.add_argument('--iter', default=100, type=int, 
                    help='Max iteration')
parser.add_argument('--conf', default=50, type=int, 
                    help='CW confidence')
parser.add_argument('--case', default=1, type=int, 
                    help='Attack generation')

parser.add_argument('--root_dir', default='/Users/eashan22/Desktop/Research/Robust ML/submission/src/',
                    type=str, help='root directory for train/evaluation')
parser.add_argument('--adv', action="store_true", default=True, 
                    help='Include Adversarial')
parser.add_argument('--eval', action="store_true", default=True, 
                    help='Testing')
parser.add_argument('--cpu', action="store_true", default=False, 
                    help='Use cpu inference')
# parser.add_argument('--topk', action="store_true", default=False, help='topk labels through cA')
args = parser.parse_args()


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No CUDA found. Working with CPU.\n")


root_dir = args.root_dir

if args.adv:
    classifier_path = str(root_dir)+"checkpoint/pgd_linf0.03/resnet50_pgd_linf_0.03_classifierA.pt"
else:
    classifier_path = str(root_dir)+"checkpoint/resnet50.pt"

model_path = str(root_dir)+"checkpoint/memory_defense_adamW.pt"


def adjust_learning_rate(learning_rate,optimizer,epoch_index,epoch):
    if epoch_index in MILESTONES:
        lr = learning_rate * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return lr


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=30):
    '''
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    '''
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer


def memory_selector(input_image):
    '''
    Classifier A: Takes input image and gives a predicted label using base classifier.
    '''
    predict_A = classifier_A(input_image)
    pred = predict_A.max(1)[1]
    
    return pred


def attackLabelGroundTruth(attack, image, label):
    '''
    Generating attack with ground truth memory label.
    Input: attack - Chosen attack for image perturbation.
           image  - Ground truth input image.
           label  - Ground truth label.
    returns: Perturbation, Adversarial label
    '''
    perturbed_image = attack.perturb(image, label)
    label_adv = memory_selector(perturbed_image)
    
    return perturbed_image, label_adv


def attackLabelClassifierA(attack, image, label):
    '''
    Generating attack with classifier A predicted memory label.
    Input: attack - Chosen attack for image perturbation.
           image  - Ground truth input image.
           label  - Predicted from Classifier-A label.
    returns: Perturbation, Adversarial label
    '''
    attack_label = memory_selector(image)
    perturbed_image = attack.perturb(image, attack_label)
    label_adv = memory_selector(perturbed_image)
    
    return perturbed_image, label_adv 


def case(argument):
    return {
        0: attackLabelGroundTruth,
        1: attackLabelClassifierA,
    }.get(argument, "Attack method does not exist!\nChoose from: \n0: ground_truth\n1: classifierA\n2: notequal") 


def get_case(CASE):
    if CASE==0:
        print("Generating attack with ground truth memory label.")
    elif CASE==1:
        print("Generating attack with classifier A predicted memory label.")
    else:
        print("Wrong case formating")
        

def load_model(model, path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


classifier_A   = resnet50(device=device)
classifier_A   = load_model(classifier_A, classifier_path)
memory_defense = MemoryDefense(args.dim, device)
memory_defense = load_model(memory_defense, model_path)


if args.adv:
    attack_dict = {
        "fgsm"    : GradientSignAttack       (memory_defense,
                                              loss_fn=None,
                                              eps=args.eps,
                                              clip_min=0.0, 
                                              clip_max=1.0,
                                              targeted=False),

        "bim"     : LinfBasicIterativeAttack (memory_defense,
                                              loss_fn=None,
                                              eps=args.eps,
                                              nb_iter=10,
                                              eps_iter=0.05,
                                              clip_min=0.0,
                                              clip_max=1.0,
                                              targeted=False),

        "pgd_linf": LinfPGDAttack            (memory_defense,
                                              loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                              eps=args.eps,
                                              nb_iter=10,
                                              eps_iter=0.0075,
                                              rand_init=True, 
                                              clip_min=0.0,
                                              clip_max=1.0,
                                              targeted=False),
        
        "cw_l2"   : CarliniWagnerL2Attack    (memory_defense,
                                              num_classes=10,
                                              confidence=args.conf,
                                              targeted=False,
                                              learning_rate=5e-3,
                                              binary_search_steps=5,
                                              max_iterations=1000,
                                              abort_early=True,
                                              initial_const=0.001,
                                              clip_min=0.0,
                                              clip_max=1.0,
                                              loss_fn=None),
        }
    
    attack = attack_dict[args.attack]
    print(f"=============================================\nAttack  : {args.attack}\nEpsilon : {args.eps+0:03}")
    print(f"=============================================\nNunber of GPU's in use: {torch.cuda.device_count()}\n=============================================\n")


transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

traindata   = torchvision.datasets.FashionMNIST(root=str(root_dir)+"/data", 
                                                train=True,
                                                download=True,
                                                transform=transform_train)
trainloader = DataLoader(traindata,
                         batch_size=args.batch, 
                         shuffle=True,
                         num_workers=8)
testdata    = torchvision.datasets.FashionMNIST(root=str(root_dir)+"/data", 
                                                train=False,
                                                download=True,
                                                transform=transform_test)
testloader  = DataLoader(testdata,
                         batch_size=args.batch,
                         shuffle=True,
                         num_workers=8)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


MSE = nn.MSELoss()
L1 = nn.L1Loss()


benign, adversarial = [], []
total1, correct1, clean_acc = 0, 0, 0
total2, correct2, attack_acc = 0, 0, 0
robust_acc, threshold, running_mem = 0.0, 0.0, 0.0
memory_labels, memory_labels_adv = {}, {}

for batch_idx, (image, label) in enumerate(testloader):
    image, label = image.to(device), label.to(device)

    # ======== ClassifierA ======== 
    pred = memory_selector(image)
    # ======== Stack ======== 
    output_dict = memory_defense(image, pred, device)
    predict, recon_x = output_dict['logit_x'], output_dict['rec_x']

    lab = torch.max(predict.data, 1)[1]
    threshold1 = MSE(image, recon_x)
    # threshold1 = MSE(image, recon_x)

    benign.append( threshold1.item() )  # ,0) )
    correct1 += (lab==label).sum().item()
    total1 += len(image)
    if args.adv:
        # Setting grad enable for gradient attacks
        torch.set_grad_enabled(True)
        # Generating the attack example using --attack method
        perturbed_image, label_adv = case(args.case)(attack, image, label)
        # Passing the perturbation and the predicted label through the defense
        output_dict = memory_defense(perturbed_image, label_adv, device)
        predict_adv, recon_x_adv = output_dict['logit_x'], output_dict['rec_x']

        lab_adv = torch.max(predict_adv.data, 1)[1]
        # Calculating the reconstruction error using mean squared error (MSE) loss.
        threshold2 = MSE(image, recon_x_adv)
        # threshold2 = MSE(image, recon_x_adv)

        # Storing mean squared error for reconstruction errors in a list.
        adversarial.append( threshold2.item() ) #,1) )
        correct2 += (lab_adv==label).sum().item()
        total2 += len(image)
        
clean_acc = (float(correct1)/total1)
if args.adv:
    attack_acc = (float(correct2)/total2)
    total = total1+total2
    correct = correct1+correct2
    robust_acc = (float(correct)/total)

    print("========================================================")
    print(f"Robust accuracy                 : {robust_acc*100:.4f}")
    print("========================================================")

if args.batch==1:
    print(f"Benign error: {min(benign)}, {max(benign)} ")
    if len(adversarial)!=0:
        print(f"Adversarial error: {min(adversarial)}, {max(adversarial)} ")

        res1 = [ele for ele in adversarial if ele > max(benign)]
        print('Total adversarial outside benign range: ', len(res1))
        res2 = [ele for ele in adversarial if ele > np.percentile(benign, 99)]
        print('Total adversarial outside  99th Percentile benign range: ', len(res2))

        total_rec = benign+adversarial
        constant = 1.0
        print('99th Percentile benign reconstruction error: ', np.percentile(benign, 99))
        print('99th Percentile adversarial reconstruction error: ', np.percentile(adversarial, 99))
    
    if len(adversarial)!=0:
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 12}
        matplotlib.rc('font', **font)
        sns.set_theme()
        
        ax = sns.distplot(benign, color="grey",  label="ben: [%.4f, %.4f], %.4s "%( min(benign), max(benign), str(np.percentile(benign,99)) ))
        sns.distplot(adversarial, color="black", label="adv: [%.4f, %.4f], %.4s "%( min(adversarial), max(adversarial), str(np.percentile(adversarial,99)) ))
        ax.set(xlabel="Reconstruction error", ylabel="No. of examples")
        
        plt.title('Attack: '+str(args.attack)+' '+str(args.eps)+', Memory Dim: '+str(args.dim))
        plt.legend()
        plt.savefig( str(root_dir)+"/"+str(args.attack)+str(args.eps)+".png", dpi=250)
        plt.show()