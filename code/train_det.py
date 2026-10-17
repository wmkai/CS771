# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from det_libs import (
    load_config,
    build_dataset,
    build_dataloader,
    FCOS,
    train_one_epoch,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)


################################################################################
def main(args):
    """main function that handles training"""

    """1. Load config / Setup folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg["output_folder"]):
        os.mkdir(cfg["output_folder"])
    cfg_filename = os.path.basename(args.config).replace(".yaml", "")
    suffix = args.output or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_folder = os.path.join(cfg['output_folder'],
                            os.path.splitext(os.path.basename(args.config))[0] + '_' + suffix)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    """2. create dataset / dataloader"""
    train_dataset = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["train"],
        cfg["dataset"]["img_folder"],
        cfg["dataset"]["json_folder"],
    )
    # data loaders
    train_loader = build_dataloader(train_dataset, True, **cfg["loader"])

    """3. create model, optimizer, and scheduler"""
    # model
    device = torch.device(cfg["devices"][0])
    model = FCOS(**cfg["model"]).to(device)
    # optimizer
    optimizer = build_optimizer(model, cfg["opt"])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)
    # mixed precision training
    scaler = None
    if cfg["dtype"] == "fp16" and device.type == "cuda":
        scaler = GradScaler()
        print("Using mixed precision training")
    # also disable cudnn benchmark, as the input size varies during training
    cudnn.benchmark = False

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(
                args.resume,
                map_location=cfg["devices"][0],
                weights_only=True
            )
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            print(
                "=> loaded checkpoint '{:s}' (epoch {:d}".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, "config.txt"), "w") as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training loop"""
    print("\nStart training ...")

    # start training
    max_epochs = cfg["opt"]["epochs"] + (cfg["opt"]["warmup_epochs"] if cfg["opt"]["warmup"] else 0)
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            torch.device(cfg["devices"][0]),
            scaler=scaler,
            tb_writer=tb_writer,
            print_freq=args.print_freq,
        )

        # save ckpt once in a while
        if ((epoch + 1) == max_epochs) or (
            (args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)
        ):
            save_states = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            }
            is_final = (epoch + 1) == max_epochs
            save_checkpoint(
                save_states,
                is_final,
                file_folder=ckpt_folder,
                file_name="epoch_{:03d}.pth.tar".format(epoch),
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return


################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train a single-stage model for object detection"
    )
    parser.add_argument("config", metavar="DIR", help="path to a config file")
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="print frequency (default: 10 iterations)",
    )
    parser.add_argument(
        "-c",
        "--ckpt-freq",
        default=1,
        type=int,
        help="checkpoint frequency (default: every 5 epochs)",
    )
    parser.add_argument(
        "--output", default="", type=str, help="name of exp folder (default: none)"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to a checkpoint (default: none)",
    )
    args = parser.parse_args()
    main(args)
