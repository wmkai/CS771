"""Train a MiniPlaces classifier using the same workflow as train_det.py."""
import argparse
import datetime
import os
from pprint import pprint

import torch
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from cls_libs import (
    load_config, build_optimizer, build_scheduler, build_model, build_loader,
    train_one_epoch, evaluate, save_checkpoint,
)


def main(args):

    """1. Load config / Setup folders"""
    cfg = load_config(args.config)
    pprint(cfg)
    suffix = args.output or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(cfg['output_folder'],
                          os.path.splitext(os.path.basename(args.config))[0] + '_' + suffix)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'config.txt'), 'w') as stream:
        pprint(cfg, stream=stream)

    # Logging using tensorboard
    writer = SummaryWriter(os.path.join(folder, 'logs'))

    """2. create dataset / dataloader"""
    device = torch.device(cfg['devices'][0])
    train_loader = build_loader(cfg, True)
    val_loader = build_loader(cfg, False)

    """3. create model, optimizer, and scheduler"""
    model = build_model(cfg).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(model, cfg['opt'])
    scheduler = build_scheduler(optimizer, cfg['opt'], len(train_loader))
    # mixed precision training
    scaler = None
    if cfg["dtype"] == "fp16" and device.type == "cuda":
        scaler = GradScaler()
        print("Using mixed precision training")
    # enable cudnn benchmark
    cudnn.benchmark = True

    """4. Resume from model / Misc"""
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint['epoch'] + 1
        print("=> resumed '{}' at epoch {}".format(args.resume, start_epoch))

    """5. training loop"""
    max_epochs = cfg['opt']['epochs'] + (
        cfg['opt']['warmup_epochs'] if cfg['opt']['warmup'] else 0)
    print('\nStart training ...')

    # the main training loop
    for epoch in range(start_epoch, max_epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                        epoch, device, scaler, tb_writer=writer, print_freq=args.print_freq)
        evaluate(val_loader, model, epoch, device,
                    tb_writer=writer, print_freq=args.print_freq)
        final = (epoch + 1 == max_epochs)
        if final or (args.ckpt_freq > 0 and (epoch + 1) % args.ckpt_freq == 0):
            save_checkpoint({
                'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            }, final, folder, 'epoch_{:03d}.pth.tar'.format(epoch))

    # wrap up
    writer.close()
    print('All done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('config', help='path to a YAML config')
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int)
    parser.add_argument('--output', default='', help='experiment folder suffix')
    parser.add_argument('--resume', default='', help='epoch checkpoint to resume')
    main(parser.parse_args())
