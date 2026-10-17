from copy import deepcopy

import yaml

# DEFAULT defines the default params used for training / inference
# the parameters here will be overwritten if a yaml config is specified
DEFAULTS = {
    # default: single gpu, can set to cpu
    "devices": ["cuda:0"],
    # fp16: to use mixed precision training 
    "dtype": "fp16",
    # output folder that stores all log files and checkpoints
    "output_folder": None,
    # CNN (custom), BASELINE / SIMPLECNN, VIT, or RESNET18
    "model_type": "CNN",
    "dataset": {
        "name": "MiniPlaces",
        # training / testing splits
        "train": "train",
        "test": "val",
        # folders that store image files / json annotations (following COCO format)
        "root_folder": None,
    },
    "loader": {
        "batch_size": 512,
        "num_workers": 4,
    },
    "input": {
        # min size: shortest side
        "img_min_size": 144,
        # crop size / color jitter / rotation (data augmentation)
        "img_crop_size": 128,
        "img_color_jitter": 0.15,
        "img_rotation": 15,
        # mean / std for input normalization
        "img_mean": [0.485, 0.456, 0.406],
        "img_std": [0.229, 0.224, 0.225],
    },
    # network architecture
    "model": {
        # number of categories; other constructor arguments may be specified here
        "num_classes": 100,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "SGD",
        # solver params
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "learning_rate": 0.1,
        # excluding the warmup epochs
        "epochs": 60,
        # if to use linear warmup
        "warmup": True,
        "warmup_epochs": 1,
        # lr scheduler: cosine / multistep
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = deepcopy(v)


def load_default_config():
    config = deepcopy(DEFAULTS)
    return config


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.safe_load(fd) or {}
    _merge(defaults, config)
    return config
