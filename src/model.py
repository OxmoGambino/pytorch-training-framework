import torch
import torchvision

import torch.nn as nn


class CNNClassif(nn.Module):

    """
    Building blocs of convolutional neural network

    Parameters :
        - nb_in_channel : Number of channel in the input image (for grayscale images : 1)
        - nb_input_linear : Size of the flattened vector before the linear layer
        - nb_channels1 : size of first convolutional layer
        - nb_channels2 : size of second convolutionnal layer
        - nb_classes : nb of classes in the dataset (10 for CIFAR-10)
    
    """
    def __init__(self,nb_in_channel, nb_input_linear,nb_channels1,nb_channels2,nb_classes):
        super().__init__()
        self.cnn_layer1 = nn.Conv2d(in_channels=nb_in_channel,out_channels=nb_channels1,kernel_size=3,padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=1)
        self.cnn_layer2 = nn.Conv2d(in_channels=nb_channels1,out_channels=nb_channels2,kernel_size=3,padding=1)
        self.cnn_linear = nn.Linear(in_features=nb_input_linear,out_features=nb_classes)

    def forward(self,x):
        x = nn.ReLU(self.cnn_layer1(x))
        x = self.max_pool(x)
        x = nn.ReLU(self.cnn_layer2(x))
        x = self.max_pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.cnn_linear(x)
        return x

