import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer :
    def __init__(self,model,train_dataloader, val_dataloader, lr=0.001,epochs=20,verbose=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #maybe one day we will have a GPU
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    
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
            predicted = self.model(images)
            loss = self.loss_fn(predicted,labels)
            loss.backward()
            self.optimizer.step()

            tr_loss += loss.item()*images.shape[0] #loss returns a mean
            # multiplication by batch_size returns the total loss of batch 
    
        
        return tr_loss / len(self.train_dataloader.dataset) 
        #division by the total number of images returns the average loss on the epoch
        #more over, the images.shape[0] handles the last batch which has often a smaller size
    
    def evaluate_model(self):
        self.model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for images,labels in self.val_dataloader :
                images, labels = images.to(self.device), labels.to(self.device) #maybe one day a gpu
                predicted = self.model(images)
                _,labels_predicted = torch.max(predicted.data,1) #each row has 10 values, the max value is the class 
                total+=labels.size(0)
                correct += (labels_predicted == labels).sum().item() #.sum() --> tensor(value) / .sum().item() --> value

        accuracy = correct/total
        return accuracy
    
    def run(self): # PENSER À IMPLÉMENTER UN EARLY STOPPING
        """Complete training with nb_epochs
        Returns :
            Last accuracy score
        """
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            acc = self.evaluate_model()
        
            if self.verbose : 
                print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss : {train_loss:.4f}, Accuracy : {acc:.4f}") #epochs+1 = beginning at 1
        return acc


