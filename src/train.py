from model import CNNClassif
from data import get_dataloaders
from trainer import Trainer
import torch
import numpy as np
import random
import wandb
from datetime import datetime


#Generate a run name according to date and hour
now = datetime.now()
run_name = f"run_test_{now.strftime('%Y%m%d_%H%M%S')}"

# Hyperparameters definition 
batch_size = 64
lr = 0.001
epochs = 25
nb_channels1 = 32
nb_channels2 = 64



def set_seed(seed): #Make the experiments reproductible ! 

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si un jour on fait tourner sur GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    wandb.init(project="pytorch-training-framework", entity="guillaume-tritz-guden-pa2", name = run_name) #initialisation as soon as we enter main
    
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
    
    wandb.finish()


if __name__ == "__main__": #lauch the training by executing python3 train.py
    main()
    