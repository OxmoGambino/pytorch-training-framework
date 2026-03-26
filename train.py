from src.model import CNNClassif
from src.data import get_dataloaders
from src.trainer import Trainer

import torch
import numpy as np
import random
import wandb
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from pathlib import Path


#Generate a run name according to date and hour
now = datetime.now()
run_name = f"run_test_{now.strftime('%Y%m%d_%H%M%S')}"

# Hyperparameters definition 
# nb_channels1 = 32
# nb_channels2 = 64



def set_seed(seed): #Make the experiments reproductible ! 

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si un jour on fait tourner sur GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    set_seed(42)
    wandb.init(project="pytorch-training-framework", #initialisation as soon as we enter main
                entity="nicoladp30-t-l-com-physique-strasbourg",
                name = run_name,
                config={
                    "lr": cfg.lr,
                    "batch_size": cfg.batch_size,
                    "epochs": cfg.epochs,
                    "nb_channels1": cfg.nb_channels1,
                    "nb_channels2": cfg.nb_channels2
                }) 

    
    print("1 - Début du script")
    print(cfg)

    train_dataloader, val_dataloader = get_dataloaders(batch_size = cfg.batch_size)
    print("2 - Dataloaders créés")

    model = CNNClassif(nb_in_channel=3,
                       nb_channels1=cfg.nb_channels1,
                       nb_channels2=cfg.nb_channels2,
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

#Affichage d'images originales avec le label prédit
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

    trainer.model.eval()

    image_batch_example, label_batch_example = next(iter(val_dataloader))

    image_batch_example = image_batch_example.to(trainer.device)
    label_batch_example = label_batch_example.to(trainer.device)

    with torch.no_grad():
        trainer.outputs = model(image_batch_example)
        predicted_labels = torch.argmax(trainer.outputs, dim=1)

    image_batch_example = image_batch_example.cpu()
    label_batch_example = label_batch_example.cpu()
    predicted_labels = predicted_labels.cpu()

    n_images = min(8, len(image_batch_example))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for ib in range(n_images):
        img = image_batch_example[ib]
        img = img / 2 + 0.5
        npimg = img.numpy().transpose((1, 2, 0))

        axes[ib].imshow(npimg)
        axes[ib].set_xticks([])
        axes[ib].set_yticks([])
        axes[ib].set_title(
            f"T: {classes[label_batch_example[ib].item()]}\n"
            f"P: {classes[predicted_labels[ib].item()]}"
        )
    print("6 - Début affichage des images")
    print("Shape batch images :", image_batch_example.shape)
    print("Labels vrais :", label_batch_example[:n_images])
    print("Labels prédits :", predicted_labels[:n_images])
    plt.tight_layout()
    save_path = Path("predictions_examples.png")
    plt.savefig(save_path)
    print(f"Image sauvegardée : {save_path.resolve()}")
    plt.show()

    wandb.finish()
    print("\n===CONFIG===")
    print("lr :", cfg.lr)
    print("batch_size :", cfg.batch_size)
    print("nb_channels1", cfg.nb_channels1)
    print("nb_channels2", cfg.nb_channels2)
    return best_acc

if __name__ == "__main__": #lauch the training by executing python3 train.py
    main()

