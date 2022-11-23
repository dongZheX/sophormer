import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
def get_optimizer(model, name="adamw", lr=0.001, weight_decay=0.0):
    if name == "adam":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, name, args, loader):
    if name == "plateau":
        # NOTE(ajayjain): For Molhiv config, this min_lr is too high -- means that lr does not decay.
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience, min_lr=args.min_lr, verbose=False)
    elif name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader), verbose=False)
    elif name == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(loader),
            pct_start=args.pct_start,
            verbose=False,
        )
    elif name is None:
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler