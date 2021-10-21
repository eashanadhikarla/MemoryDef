from __future__ import absolute_import, print_function

from models import MemoryDefense
from resnet import resnet50
# from vanillaAE import VAE
from util import *

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import os, pickle, math, logging, random, time

from sklearn.manifold import TSNE
import matplotlib, seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# %matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "~/src/"

BATCH_SIZE = 100
MEM_DIM=500
MEMORY_DEFENSE = True

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
mnist_test  = torchvision.datasets.FashionMNIST(
    root=str(root_dir)+"/data",
    train=False,
    transform=transform_test
    )
test_loader = DataLoader(
    mnist_test,
    batch_size=BATCH_SIZE,
    shuffle=False
    )

classifierA = resnet50(device=device)
classifierA.load_state_dict(
    torch.load(str(root_dir)+"checkpoint/resnet50.pt",
    map_location=torch.device('cpu'))
    )
classifierA.to(device)
classifierA.eval()

if MEMORY_DEFENSE:
    model = MemoryDefense(MEM_DIM, device)
    model.load_state_dict(
        torch.load(str(root_dir)+"checkpoint/memory_defense_adamW.pt",
        map_location=torch.device('cpu'))
        )
else:
    model = VAE(device)
    model.load_state_dict(torch.load(str(root_dir)+"checkpoint/vae.pt"))
model.to(device)
model.eval()

# def load_model(model, path):
#     state_dict = torch.load(path, map_location=torch.device('cpu'))
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model


# classifier_A   = resnet50(device=device)
# classifier_A   = load_model(classifier_A, classifier_path)
# memory_defense = MemoryDefense(args.dim, device)
# memory_defense = load_model(memory_defense, model_path)

colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])

def scatter(x, colors):
    '''
    https://github.com/breandan/ml-exercises/blob/master/tSNE.py
    
    '''
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))
    
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=5,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


optimizer = torch.optim.Adam(model.parameters())

total_arr = []
for idx,[image,label] in enumerate(test_loader):
    x, y = image.to(device), label.to(device)
    
    predict_A = classifierA(x)
    pred = predict_A.max(1)[1]
    
    optimizer.zero_grad()
    output = model.forward(x, pred, device)
    encoded = output['encoded']
    
    for k in range(BATCH_SIZE):
        total_arr.append(encoded[k].view(-1).cpu().data.numpy())

    # if idx>125:
    #     break

tsne_model = TSNE(n_components=2, init='pca',random_state=0)
result = tsne_model.fit_transform(total_arr)

label=[]
for i in range(len(result)):
    label.append(mnist_test[i][1])
    
lab = np.asarray(label)
scatter(result, lab)
plt.savefig(str(root_dir)+'tsne-fmnist.png', dpi=150)