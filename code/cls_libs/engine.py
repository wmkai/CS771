import time

import torch

from .utils import AverageMeter, accuracy


def train_one_epoch(
    train_loader, 
    model, 
    criterion, 
    optimizer, 
    scheduler, 
    curr_epoch, 
    device,
    scaler=None, 
    tb_writer=None,
    print_freq=10
):
    """Training the model"""
    # adjust the learning rate
    num_iters = len(train_loader)

    # set up meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for iter_idx, (imgs, targets) in enumerate(train_loader, 0):
        # zero grad
        optimizer.zero_grad()

        # data -> GPU
        imgs = imgs.to(device)
        targets = targets.to(device)

        # compute output
        if scaler != None:
            # mixed precision training
            # forward / backward the model
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            # step optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # update the scaler
            scaler.update()
            # step the scheduler
            scheduler.step()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            # compute gradient and do one SGD step 
            loss.backward()
            optimizer.step()
            # step the scheduler
            scheduler.step()

        # printing
        if (iter_idx != 0) and print_freq > 0 and (iter_idx % print_freq) == 0:
            # measure elapsed time
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_time.update((time.time() - end) / print_freq)
            end = time.time()

            # measure accuracy and record loss (from current batch)
            # stats are gathered at every few iterations to save time
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1.item(), imgs.size(0))
            top5.update(acc5.item(), imgs.size(0))

            # print the stats
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.2f} ({loss.avg:.2f})\t"
                "Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t"
                "Acc@5 {top5.val:.2f} ({top5.avg:.2f})".format(
                    curr_epoch,
                    iter_idx,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

            # log to tensorboard
            if tb_writer is not None:
                # log loss / lr
                tb_writer.add_scalar(
                    "data/training_loss", 
                    losses.val, 
                    curr_epoch * num_iters + iter_idx
                )
                tb_writer.add_scalar(
                    "data/learning_rate",
                    scheduler.get_last_lr()[0],
                    curr_epoch * num_iters + iter_idx
                )

    # print the learning rate
    print(
        "[Training]: Epoch {:d} finished with lr={:f}".format(
            curr_epoch, scheduler.get_last_lr()[0]
        )
    )

    if tb_writer is not None:
        # log top-1/5 acc
        tb_writer.add_scalars("data/top1_accuracy", {"train": top1.avg}, curr_epoch + 1)
        tb_writer.add_scalars("data/top5_accuracy", {"train": top5.avg}, curr_epoch + 1)


@torch.no_grad()
def evaluate(
    val_loader, 
    model, 
    curr_epoch,
    device,
    tb_writer=None,
    eval_aug=False,
    print_freq=10
):
    """Test the model on the validation set"""
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Disable training behavior; the decorator disables autograd.
    model.eval()
    end = time.time()

    # loop over validation set
    for iter_idx, (imgs, targets) in enumerate(val_loader, 0):
        
        # data -> GPU
        imgs = imgs.to(device)
        targets = targets.to(device)

        # compute output
        outputs = model(imgs)

        if eval_aug:
            flipped_imgs = torch.flip(imgs, (3,))
            flipped_outputs = model(flipped_imgs)
            outputs = 0.5 * (outputs + flipped_outputs)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1.item(), imgs.size(0))
        top5.update(acc5.item(), imgs.size(0))
        
        # measure elapsed time
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_time.update(time.time() - end)
        end = time.time()

        # printing
        if (iter_idx != 0) and print_freq > 0 and (iter_idx % print_freq) == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t"
                "Acc@5 {top5.val:.2f} ({top5.avg:.2f})".format(
                    iter_idx, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5
                )
            )

    print(
        "******Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
            top1=top1, top5=top5
        )
    )

    if tb_writer is not None:
        # log top-1/5 acc
        tb_writer.add_scalars("data/top1_accuracy", {"val": top1.avg}, curr_epoch + 1)
        tb_writer.add_scalars("data/top5_accuracy", {"val": top5.avg}, curr_epoch + 1)

    return top1.avg


