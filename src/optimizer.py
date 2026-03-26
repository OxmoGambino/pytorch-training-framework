import torch


def build_optimizer(model, cfg):
    """
    Construit l'optimiseur à partir de la config Hydra
    """
    optimizer_name = cfg.optimizer.name.lower()
    lr = cfg.lr
    weight_decay = cfg.optimizer.weight_decay

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=weight_decay
        )

    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=cfg.optimizer.momentum
        )

    else:
        raise ValueError(f"Optimiseur inconnu : {cfg.optimizer.name}")