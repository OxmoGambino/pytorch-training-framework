from datetime import datetime

def build_run_name(cfg): 
    """
    Noms plus clairs pour le run
    """
    model = cfg.model.name.lower()
    optim = cfg.optimizer.name.lower()
    loss = cfg.loss.name.lower()
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    name = f"{model} {optim} {loss} {time}"
    
    return name

    