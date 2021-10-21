from __future__ import absolute_import
from torchvision.transforms import *

from torch.utils.data import TensorDataset
import torch, math, random, io, requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root_dir = "/home/jupyter-eaa418/src/fmnist"

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean = [0.4914]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

def preprocessing(key, epsilon, training="stack_adversarial"):
    '''
        returns dataset: shuffles benign and adversarial and can be fed directly to dataloader.
    '''
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
            traindata = dataset
        elif opt=='test':
            testdata  = dataset
    print('Adversarial data loaded successfully.')
    return traindata, testdata

def batch_inverse_normalize(x, mean, std):
    # represent mean and std to 1, C, 1, ... tensors for broadcasting
    reshape_shape = [1, -1] + ([1] * (len(x.shape) - 2))
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).reshape(*reshape_shape)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).reshape(*reshape_shape)
    return x / std + mean

def inv_normalize(img):
    mean = torch.Tensor([0.1307]).unsqueeze(-1)
    std= torch.Tensor([0.3081]).unsqueeze(-1)
    img = (img.view(3, -1) * std + mean).view(img.shape)
    img = img.clamp(0, 1)
    return img

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot(input_np, ys, recon_np, pred_ys):
    '''
        Input: Numpy images to be plotted.
    '''
    fig, ax = plt.subplots(1,2)
    im1 = ax[0].imshow(np.squeeze(input_np.numpy()[0]), cmap='gray', interpolation='none')
    ax[0].title.set_text('Input  - '+str(int(ys[0].cpu().detach().numpy().item())))
    im2 = ax[1].imshow(np.squeeze(recon_np.numpy()[0]), cmap='gray', interpolation='none')
    ax[1].title.set_text('Output - '+str(int(pred_ys[0].cpu().detach().numpy().item())))
    plt.show()

def load_data(list_):
    '''
        Loads the data in the form of Torch Dataloader.
        Input: Dataset {type = list()}
    '''
    if len(list_[0])==2:
        img, lab = [], []
        for i in list_:
            img.append(i[0]), lab.append(i[1])
        xs = torch.stack(img)
        xs = xs.squeeze(1)
        ys = torch.Tensor(lab)
        dataset = TensorDataset(xs, ys)
        del img, lab
    elif len(list_[0])==3:
        img, lab, flag = [], [], []
        for i in list_:
            img.append(i[0]), lab.append(i[1]), flag.append(i[2])
        xs = torch.stack(img)
        xs = xs.squeeze(1)
        ys = torch.Tensor(lab)
        flag = torch.Tensor(flag)
        dataset = TensorDataset(xs, ys, flag)
        del img, lab, flag

    return dataset

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


