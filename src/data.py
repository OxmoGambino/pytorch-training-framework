import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1),transforms.ToTensor()])
#Gestion des 3 canaux faites en passant en grayscale (1 canal)
#Voir plus tard pour gérer image en couleurs avec Deepwise puis Pointwise Convolution


trainset = torchvision.datasets.CIFAR10(root='..\data',train=True,download=False,transform=transform)