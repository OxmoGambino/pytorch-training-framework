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

def build_train_transform(cfg):
    """
    Construit dynamiquement la pipeline de transformations
    du train set à partir de la config Hydra.
    """
    transform_list=[]

    if (cfg.augmentation.enabled):
        if (cfg.augmentation.random_crop.enabled):
            transform_list.append(transforms.RandomCrop(size=cfg.augmentation.random_crop.size,
                                 padding=cfg.augmentation.random_crop.padding))

        if (cfg.augmentation.horizontal_flip.enabled):
            transform_list.append(transforms.RandomHorizontalFlip(
                                    p=cfg.augmentation.horizontal_flip.p))

        if (cfg.augmentation.rotation.enabled):
            transform_list.append(transforms.RandomRotation(
                                    degrees=cfg.augmentation.rotation.degrees))

    transform_list.extend([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return transforms.Compose(transform_list)


def build_val_transform():
    """
    Pas d’augmentation pour la validation/test.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])







def get_dataloaders(cfg): 
    """Generate trainng and validation dataloaders on CIFAR-10

    Args:
        batch_size : Default to 64
    
    Returns:
        Tuple (train_dataloader, val_dataloader)
    """
    train_transform = build_train_transform(cfg)
    val_transform = build_val_transform()

    trainset = torchvision.datasets.CIFAR10( root="./data",
                                            train=True,
                                            download=False,
                                            transform=train_transform)
    
    valset = torchvision.datasets.CIFAR10(root="./data",
                                        train=False,
                                        download=False,
                                        transform=val_transform)
    
    trainloader = DataLoader( trainset,
                            batch_size=cfg.training.batch_size,
                            shuffle=True)
    
    valloader = DataLoader(valset,
                        batch_size=cfg.training.batch_size,
                        shuffle=False)

    return trainloader, valloader
