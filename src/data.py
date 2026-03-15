import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pathlib import Path

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


    transform = transforms.Compose([transforms.ToTensor()]) #look for normalization (see Deep Learning Lectures)

    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR,
                                            train=True,
                                            download=True,
                                            transform=transform) #load data_batch1 -> data_batch5
    
    train_set = Subset(train_set, range(5000)) #Moins gourmand à l'entrainement sur CPU
    
    val_set = torchvision.datasets.CIFAR10(root=DATA_DIR,
                                            train=False,
                                            download=True,
                                            transform=transform) #load test_batch
    
    train_dataloader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_dataloader = DataLoader(val_set,batch_size = batch_size,shuffle=False)

    return train_dataloader, val_dataloader