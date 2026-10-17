import os

import torch
import torchvision

import torch.optim as optim
from torch.utils.data import DataLoader

from .lr_scheduler import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .dataset import MiniPlacesLoader
from .modeling import SimpleNet, default_cnn_model, default_vit_model
from .transforms import get_train_transforms, get_val_transforms


def save_checkpoint(state, is_final, file_folder, file_name="checkpoint.pth.tar"):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_final:
        # skip the optimization / scheduler state
        state = {k: v for k, v in state.items() if k not in ("optimizer", "scheduler")}
        torch.save(state, os.path.join(file_folder, "model_final.pth.tar"))


def build_optimizer(model, optimizer_config):
    """
    Return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    blacklist_weight_modules = (torch.nn.BatchNorm2d, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if p.ndim <= 1 or pn.endswith("bias") or "pos_embed" in pn:
                # all biases / position embedding will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": optimizer_config["weight_decay"],
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"],
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(optim_groups, lr=optimizer_config["learning_rate"])
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def build_scheduler(optimizer, optimizer_config, num_iters_per_epoch, last_epoch=-1):
    """
    Return a supported learning rate scheduler
    All schedulers returned by this function should step every iteration
    """
    if optimizer_config["warmup"] and optimizer_config["warmup_epochs"] > 0:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_steps, max_steps, last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [
                num_iters_per_epoch * step
                for step in optimizer_config["schedule_steps"]
            ]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch,
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max_steps, last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [
                num_iters_per_epoch * step
                for step in optimizer_config["schedule_steps"]
            ]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch,
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


def build_model(cfg):
    """Build the configured model."""
    model_type = cfg['model_type'].upper()
    options = dict(cfg['model'])
    if model_type in ('BASELINE', 'SIMPLECNN'):
        return SimpleNet(**options)
    if model_type == 'CNN':
        return default_cnn_model(**options)
    if model_type == 'VIT':
        options.setdefault("img_size", cfg["input"]["img_crop_size"])
        return default_vit_model(**options)
    if model_type == 'RESNET18':
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Linear(512, cfg['model']['num_classes'])
        return model
    raise ValueError('Unsupported model_type: ' + cfg['model_type'])


def build_loader(cfg, training):
    """Build the dataloader."""
    if cfg['dataset']['name'] != 'MiniPlaces':
        raise ValueError('Only MiniPlaces is supported')
    dataset = MiniPlacesLoader(
        cfg['dataset']['root_folder'],
        split=cfg['dataset']['train' if training else 'test'],
        num_classes=cfg['model']['num_classes'],
        transforms=(get_train_transforms if training else get_val_transforms)(cfg['input']),
    )
    if not len(dataset):
        raise ValueError('Dataset is empty')
    return DataLoader(dataset, shuffle=training, **cfg['loader'])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
