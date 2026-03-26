import torch.nn.functional as F #useful for forward pass
import torch.nn as nn
import torchvision.models as models

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
    
    def __init__(self,nb_in_channel,nb_channels1,nb_channels2,nb_classes):
        super().__init__()
        self.cnn_layer1 = nn.Conv2d(in_channels=nb_in_channel,out_channels=nb_channels1,kernel_size=3,padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2) #stride = 2 with kernel = 2 : divide size by 2 #BUT WHY?
        self.cnn_layer2 = nn.Conv2d(in_channels=nb_channels1,out_channels=nb_channels2,kernel_size=3,padding=1)
        self.cnn_linear = nn.Linear(in_features=nb_channels2*8*8,out_features=nb_classes) ## Voir si on peut pas faire un calcul automatique si on change les paramètres

    def forward(self,x):
        x = F.relu(self.cnn_layer1(x)) #la syntaxe c'est pas ReLU ?
        x = self.max_pool(x)
        x = F.relu(self.cnn_layer2(x))
        x = self.max_pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.cnn_linear(x)
        return x


class MLPClassif(nn.Module):
    def __init__(self, input_dim=32*32*3, hidden_dim=512, nb_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),nn.Linear(input_dim,hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, nb_classes))
        
    def forward(self, x):
        return self.net(x)



class ResNet18Classif(nn.Module):
    def __init__(self, nb_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, nb_classes)

    def forward(self, x):
        return self.model(x)
    




def build_model(cfg):
    model_name = cfg.model.name.lower()

    if (model_name == "mlp"):
        return MLPClassif(hidden_dim=cfg.model.mlp.hidden_dim,
                          nb_classes=cfg.model.nb_classes)
    
    elif (model_name == "cnn"):
        return CNNClassif(nb_in_channel=3,
                           nb_channels1=cfg.model.cnn.nb_channels1,
                           nb_channels2=cfg.model.cnn.nb_channels2,
                           nb_classes=cfg.model.nb_classes)
    

    elif (model_name == 'resnet18'):
        return ResNet18Classif(nb_classes=cfg.model.nb_classes)
    
    else:
        raise ValueError(f"Uknown model name : {model_name}")