import torch 


def build_loss(cfg): #regarder plus de loss si possible ? 
    name = cfg.loss.name.lower() 
    if name == "cross_entropy": #classique 99% des pb de classification
        return torch.nn.CrossEntropyLoss()
    
    elif name=="multi_margin":
        return torch.nn.MultiMarginLoss()
        
    raise ValueError(f"Unknown loss : {cfg.loss.name}") #au cas ou une loss n'est pas dispo dans le build