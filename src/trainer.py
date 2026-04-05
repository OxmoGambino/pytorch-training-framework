import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import wandb
import torch.optim as optim
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class Trainer :
    def __init__(self,model,train_dataloader, val_dataloader, optimizer,loss_fn,epochs=25,verbose=True, early_stopping=True, patience=5, min_delta=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #maybe one day we will have a GPU (no we poor)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.verbose = verbose  
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

        #Early Stopping Parameters
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        #scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                mode='min',
                                                                factor=0.5,
                                                                patience=2)

    def save_checkpoint(self, epoch, best_val_loss):
        checkpoint = {"epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "best_val_loss": best_val_loss}
        torch.save(checkpoint, self.checkpoint_path)
    
    def train_one_epoch(self):
        """Do one epoch training
        Returns: 
            Loss on the epoch (float)
        """
        self.model.train() 
        tr_loss = 0

        for images,labels in self.train_dataloader:
            images, labels = images.to(self.device), labels.to(self.device) #myabe one day a gpu
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs,labels)
            loss.backward()
            self.optimizer.step()

            tr_loss += loss.item()*images.shape[0] #loss returns a mean
            # multiplication by batch_size returns the total loss of batch 
    
        
        return tr_loss / len(self.train_dataloader.dataset) 
        #division by the total number of images returns the average loss on the epoch
        #more over, the images.shape[0] handles the last batch which has often a smaller size
    


    def evaluate_model(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images,labels in self.val_dataloader :
                images, labels = images.to(self.device), labels.to(self.device) #maybe one day a gpu
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                val_loss += loss.item() * images.shape[0]

                _,labels_predicted = torch.max(outputs.data,1) #each row has 10 values, the max value is the class 
                total+=labels.size(0)
                correct += (labels_predicted == labels).sum().item() #.sum() --> tensor(value) / .sum().item() --> value



        val_loss = val_loss / len(self.val_dataloader.dataset)
        accuracy = correct/total
        return val_loss, accuracy
    


    def run(self): # PENSER À IMPLÉMENTER UN EARLY STOPPING (c'est fait hahahahahahahah)
        """Complete training with nb_epochs
        Returns :
            Last accuracy score
        """
        best_val_loss = float("inf")
        best_acc = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss, acc = self.evaluate_model()
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            current_weight_decay = self.optimizer.param_groups[0]['weight_decay']
            print(f"LR actuel : {current_lr}")
            
            if self.verbose : 
                print(f"Epoch [{epoch+1}/{self.epochs}],"
                      f"Training Loss : {train_loss:.4f},"
                      f"Validation loss : {val_loss: .4f}, "
                      f"Accuracy : {acc:.4f},"
                      f"Learning Rate : {current_lr},"
                      f"Weight Decay : {current_weight_decay}") #epochs+1 = beginning at 1
                
            if (acc > best_acc) : 
                best_acc = acc #Save best accuracy seen

            if (val_loss < best_val_loss - self.min_delta):
                best_val_loss = val_loss #Save best val_loss seen
                patience_counter = 0
                self.save_checkpoint(epoch=epoch,
                                    best_val_loss=best_val_loss)
                best_model_state = copy.deepcopy(self.model.state_dict()) #save a true copy of real weights
            else:
                patience_counter += 1

            if (self.verbose):
                print(f"No improvment for {patience_counter} epoch(s).")

            if (self.early_stopping and patience_counter >= self.patience):
                if (self.verbose):
                    print("Early stopping triggered.")
                break
            
            wandb.log({"epoch" : epoch+1,
                        "train_loss" : train_loss,
                        "val_loss" : val_loss,
                        "val_acc" : acc,
                        "best_val_accuracy_yet" : best_acc,
                        "patience_counter" : patience_counter}) #ça peut être intéressant la patience de l'early stopping
            

        if (best_model_state is not None):
            self.model.load_state_dict(best_model_state)

        return best_acc


