import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np


""" Loading and preprocessing of the dataset CIFAR-10"""


ROOT = Path(__file__).parent.parent  # src/ → pytorch-training-framework/
DATA_DIR = ROOT / "data"             # p-t-f/data/

def get_dataloaders(batch_size = 64): #batch size of 64 is great for CPU training (not too large but not too small.)
    """Generate trainng and validation dataloaders on CIFAR-10

    Args:
        batch_size : Default to 64
    
    Returns:
        Tuple (train_dataloader, val_dataloader)
    """
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]) #look for normalization (see Deep Learning Lectures)
    

    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR,
                                            train=True,
                                            download=True,
                                            transform=train_transform) #load data_batch1 -> data_batch5
    
    
    # Application d'un subset stratifié de 5000 images (500 images par classes)
    targets = np.array(train_set.targets) 
    indices = np.arange(len(targets))
    subset_indices,_ = train_test_split(indices, train_size=0.1,stratify=targets,random_state=42)
    
  
    
    #subset_targets = targets[subset_indices]
    #print(Counter(subset_targets))
    
    train_set = Subset(train_set,indices=subset_indices) #Moins gourmand à l'entrainement sur CPU 
    #500 images par classes


    
    
    
    
    val_set = torchvision.datasets.CIFAR10(root=DATA_DIR,
                                            train=False,
                                            download=True,
                                            transform=val_transform) #load test_batch
    
    train_dataloader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_dataloader = DataLoader(val_set,batch_size = batch_size,shuffle=False)

    return train_dataloader, val_dataloader