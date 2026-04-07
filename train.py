from src.data import get_dataloaders
from src.trainer import Trainer

import torch
import numpy as np
import random
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from pathlib import Path
from src.optimizer import build_optimizer
from src.model import build_model
from src.loss import build_loss
from src.utils import build_run_name


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
    run_name = build_run_name(cfg)
    
    wandb.init(project="pytorch-training-framework", #initialisation as soon as we enter main
                entity="nicoladp30-t-l-com-physique-strasbourg",
                name = run_name,
                config=OmegaConf.to_container(cfg, resolve=True)) 
                #on prend tous les param du .yaml, pas d'oublis et pas besoin de modifier l'init à chaque modif du .yaml

    
    print("1 - Début du script")
    print(cfg)

    train_dataloader, val_dataloader = get_dataloaders(cfg)
    print("2 - Dataloaders créés")

    model = build_model(cfg) #Création du modèle selon le modèle choisit à l'exécution

    print("3 - Modèle créé")

    optimizer = build_optimizer(model, cfg)
    loss_fn = build_loss(cfg)
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=cfg.training.epochs,
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
    print("lr :", cfg.training.lr)
    print("batch_size :", cfg.training.batch_size)
    print("nb_channels1", cfg.model.cnn.nb_channels1)
    print("nb_channels2", cfg.model.cnn.nb_channels2)
    return best_acc

if __name__ == "__main__": #lauch the training by executing python3 train.py
    main()

