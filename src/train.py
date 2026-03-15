from model import CNNClassif
from data import get_dataloaders
from trainer import Trainer

# Hyperparameters definition 
batch_size = 64
lr = 0.001
epochs = 20
nb_channels1 = 32
nb_channels2 = 64

if __name__ == "__main__": #lauch the training by executing python3 train.py
    train_dataloader, val_dataloader = get_dataloaders(batch_size = batch_size)

    model = CNNClassif(nb_in_channel=3,
                       nb_channels1=nb_channels1,
                       nb_channels2=nb_channels2,
                       nb_classes=10)
    
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      lr=lr,
                      epochs=epochs,
                      verbose=True
    )

    acc = trainer.run()
    print(f"\nFinal Accuracy : {acc:.4f}")
    