from src.model import CNNClassif
from src.data import get_dataloaders
from src.trainer import Trainer

import torch
import numpy as np
import random
import wandb
from datetime import datetime
import hydra
from omegaconf import DictConfig

#Generate a run name according to date and hour
now = datetime.now()
run_name = f"run_test_{now.strftime('%Y%m%d_%H%M%S')}"

# Hyperparameters definition 
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

@hydra.main(config_path="conf",config_name="config")
def main(cfg : DictConfig):
    set_seed(42)
    wandb.init(project="pytorch-training-framework", entity="nicoladp30-t-l-com-physique-strasbourg", name = run_name) #initialisation as soon as we enter main
    
    print("1 - Début du script")
    print(cfg)

    train_dataloader, val_dataloader = get_dataloaders(batch_size = cfg.batch_size)
    print("2 - Dataloaders créés")

    model = CNNClassif(nb_in_channel=3,
                       nb_channels1=nb_channels1,
                       nb_channels2=nb_channels2,
                       nb_classes=10)
    print("3 - Modèle créé")
    
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      lr=cfg.lr,
                      epochs=cfg.epochs,
                      verbose=True
    )
    print("4 - Trainer créé")

    best_acc = trainer.run()
    print("5 - run() terminé")
    print(f"\nBest validation Accuracy : {best_acc:.4f}")
    
    wandb.finish()


if __name__ == "__main__": #lauch the training by executing python3 train.py
    main()
    