from model import CNNClassif
from data import get_dataloaders
from trainer import Trainer
import torch
import numpy as np
import random

def set_seed(seed): #Make the experiments reproductible ! 

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si un jour on fait tourner sur GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparameters definition 
batch_size = 64
lr = 0.001
epochs = 25
nb_channels1 = 32
nb_channels2 = 64

if __name__ == "__main__": #lauch the training by executing python3 train.py
    set_seed(42)
    print("1 - Début du script")

    train_dataloader, val_dataloader = get_dataloaders(batch_size = batch_size)
    print("2 - Dataloaders créés")

    model = CNNClassif(nb_in_channel=3,
                       nb_channels1=nb_channels1,
                       nb_channels2=nb_channels2,
                       nb_classes=10)
    print("3 - Modèle créé")
    
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      lr=lr,
                      epochs=epochs,
                      verbose=True
    )
    print("4 - Trainer créé")

    best_acc = trainer.run()
    print("5 - run() terminé")
    print(f"\nBest validation Accuracy : {best_acc:.4f}")
    