from distutils.command.build import build
from pathlib import Path
import argparse
import os
import sys
import random

import time
import json
import math
import numpy as np

from torch import nn, optim
import torch
import torch.distributed as dist
import torchvision
from PIL import Image

import tensorboard_logger as tb_logger

import torchvision.transforms as transforms

from utils import gather_from_all, GaussianBlur, Solarization

# from datasets import build_dataset
# from memory import build_mem

# def main():
#     args = parser.parse_args()
#     args.ngpus_per_node = torch.cuda.device_count()
#     args.scale = [float(x) for x in args.scale.split(',')]
#     if 'SLURM_JOB_ID' in os.environ:
#         cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
#         stdout = subprocess.check_output(cmd.split())
#         host_name = stdout.decode().splitlines()[0]
#         args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
#         args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
#         args.dist_url = f'tcp://{host_name}:58478'
#     else:
#         # single-node distributed training
#         args.rank = 0
#         args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
#         args.world_size = args.ngpus_per_node
#     torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_worker(gpu, args):

    fix_seed(args.seed)

    if args.ddp:
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

        logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = SimCLR(args).cuda(gpu)
    if args.ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=exclude_bias_and_norm,
                        lars_adaptation_filter=exclude_bias_and_norm)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0,
                                    momentum=args.opt_momentum, weight_decay=args.weight_decay)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    else:
        start_epoch = 0

    if args.supcon_new:
        dataset = torchvision.datasets.ImageFolder(args.data / 'train', Single_Transform(args))
    else:
        dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform(args))
    
    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=per_device_batch_size, num_workers=args.workers,
            pin_memory=True, sampler=sampler)
    else:
        sampler = None
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, drop_last=True)
        
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.ddp:
            sampler.set_epoch(epoch)

        for step, (data, labels) in enumerate(loader, start=epoch * len(loader)):
            if args.supcon_new:
                y1 = data
                y2, y3 = None, None
            else:   
                (y1, y2, y3) = data
                y2 = y2.cuda(gpu, non_blocking=True)
                
            y1 = y1.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            if args.rotation:
                if args.supcon_new:
                    raise NotImplementedError
                y3 = y3.cuda(gpu, non_blocking=True)
                rotated_images, rotated_labels = rotate_images(y3, gpu)

            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss, acc = model.forward(y1, y2=y2, labels=labels)

                if args.rotation:
                    logits = model.module.forward_rotation(rotated_images)
                    rot_loss = torch.nn.functional.cross_entropy(logits, rotated_labels)
                    loss += args.rotation * rot_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.print_freq == 0:
                if args.ddp:
                    torch.distributed.reduce(acc.div_(args.world_size), 0)
                if args.rank == 0:
                    print(f'epoch={epoch}, step={step}, loss={loss.item()}, acc={acc.item()}', flush=True)
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), acc=acc.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                            optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

            # save checkpoint to epoch
            if epoch % args.save_freq == 0 or epoch < 10:
                torch.save(state, args.checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch))

            # log to tensorboard
            logger.log_value('loss', loss.item(), epoch)
            logger.log_value('acc', acc.item(), epoch)
            logger.log_value('learning_rate', lr, epoch)

    if args.rank == 0:
        # save final model
        torch.save(dict(backbone=model.module.backbone.state_dict(),
                        projector=model.module.projector.state_dict(),
                        head=model.module.online_head.state_dict()),
                args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate #* args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.dataset == 'imagenet':
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        elif args.dataset == 'cifar100':
            self.backbone = torchvision.models.resnet18(zero_init_residual=True)

        self.backbone.fc = nn.Identity()

        # projector
        if args.dataset == 'imagenet':
            sizes = [2048, 2048, 2048, 3]
        elif args.dataset == 'cifar100':
            sizes = [512, 512, 512, 128]

        args.feat_dim = sizes[-1]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))
        self.projector = nn.Sequential(*layers)

        if args.dataset == 'imagenet':
            self.online_head = nn.Linear(2048, 1000)
        elif args.dataset == 'cifar100':
            self.online_head = nn.Linear(512, 100)

        if args.rotation:
            self.rotation_projector = nn.Sequential(nn.Linear(2048, 2048),
                                                    nn.LayerNorm(2048),
                                                    nn.ReLU(inplace=True),  # first layer
                                                    nn.Linear(2048, 2048),
                                                    nn.LayerNorm(2048),
                                                    nn.ReLU(inplace=True),  # second layer
                                                    nn.Linear(2048, 128),
                                                    nn.LayerNorm(128),
                                                    nn.Linear(128, 4))  # output layer


    def forward(self, y1, y2=None, labels=None):
        r1 = self.backbone(y1)
        # projection
        z1 = self.projector(r1)

        if not self.args.supcon_new:
            r2 = self.backbone(y2)
            z2 = self.projector(r2)
        else:
            z2 = None     
     
        if self.args.supcon_new:
            loss = infoNCE_supcon_new(z1, labels, thresh=self.args.threshold)
        else:
            loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2
        # elif self.args.mask_mode == 'pos':
        #     loss = infoNCE_pos(z1, z2, labels, self.args.temp) / 2 + infoNCE_pos(z2, z1, labels, self.args.temp) / 2
        # elif self.args.mask_mode == 'supcon':
        #     loss = infoNCE_supcon(z1, z2, labels, self.args.temp) / 2 + infoNCE_supcon(z2, z1, labels, self.args.temp) / 2
        # elif self.args.mask_mode == 'supcon_all':
        #     loss = infoNCE_supcon_all(z1, z2, labels, self.args.temp) / 2 + infoNCE_supcon_all(z2, z1, labels, self.args.temp) / 2

        logits = self.online_head(r1.detach())
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        loss = loss + cls_loss

        return loss, acc

    def forward_rotation(self, x):
        b = self.backbone(x)
        logits = self.rotation_projector(b)

        return logits

def infoNCE(nn, p, temperature=0.2, gather_all=True):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    if gather_all:
        nn = gather_from_all(nn)
        p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def infoNCE_supcon(nn, p, labels, temperature=0.2, gather_all=True):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    if gather_all:
        nn = gather_from_all(nn)
        p = gather_from_all(p)
        labels = gather_from_all(labels)

    n = p.shape[0]
    labels = labels.view(-1, 1)
    labels_cat = labels.repeat(2,1)

    p = torch.cat([nn,p], dim=0)
    logits = nn @ p.T
    logits /= temperature

    # mask for positive
    pos_mask = torch.eq(labels, labels_cat.transpose(1,0)).float().cuda(non_blocking=True)
    pos_mask.fill_diagonal_(0.0)

    # mask for positive (only other view) + negatives
    pos_neg_mask = (~torch.eq(labels, labels.transpose(1,0))).float().cuda(non_blocking=True) + torch.eye(labels.shape[0]).float().cuda(non_blocking=True)
    denom_exp_logits = torch.exp(logits) * pos_neg_mask

    log_prob = -1.0 * (logits - torch.log(denom_exp_logits.sum(dim=1, keepdim=True)))
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(dim=1)

    return mean_log_prob_pos.mean()

def infoNCE_supcon_new(features, labels, thresh=0.0, include_diag=True, same_class_negs=True, contrast_mode='all', temperature=0.07, base_temperature=0.07, gather_all=True):
    if gather_all:
        features = gather_from_all(features)
        labels = gather_from_all(labels)
    # add second_dimension (For compatibiility)
    features = torch.unsqueeze(features, 1)
    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)
    batch_size = features.shape[0]
    if labels is not None:
        labels = labels.contiguous()
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        # mask = torch.eq(labels, labels.T).float().to(device)
    else:
        raise TypeError("Labels cannot be None.")
    
    # normalize features
    features = torch.nn.functional.normalize(features, dim=2)
    contrast_count = features.shape[1] # = n_views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))
    
    contrast_labels = labels.repeat(contrast_count).contiguous()
    anchor_labels = labels.repeat(anchor_count).contiguous()
    # _, a_idx = torch.sort(anchor_labels, stable = True)
    # _, inv_idx = torch.sort(a_idx, stable = True)
    _, a_idx = torch.sort(anchor_labels)
    _, inv_idx = torch.sort(a_idx)
    
    # compute logits
    # anchor_dot_contrast and logits are sorted by label along dimension 0
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # num_labels = torch.unique(labels).max() + 1
    
    # calculating percentiles for positive pairs
    # label_percentile_dists = torch.zeros(num_labels).detach()
    # masks for values in the same class
    mask = torch.eq(anchor_labels.view(-1,1), contrast_labels.view(1,-1)).cuda(non_blocking=True)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).cuda(non_blocking=True),
        0
    )
    #print("removing diagonal mask")
    if not include_diag:
        mask = mask * logits_mask # selects elements of the same class and not along diagonal
    # aug_mask is tiled identity matrices, no need to remove main diagonal
    # because it gets zeroed out when multiplied by logits_mask
    # aug_mask = torch.eye(batch_size).tile((anchor_count, contrast_count)).cuda(non_blocking=True)
    # tile doesnt exist in pytorch 1.7.1, so we use repeat
    aug_mask = torch.eye(batch_size).repeat(anchor_count, contrast_count).cuda(non_blocking=True)
    
    # offset version of anchor_dot_contrast that masks diagonal entries (self) and not in same class
    # and removes its own augmented views
    temp = ((anchor_dot_contrast + 2/temperature) * mask  * (1 - aug_mask))
    
    sorted_temp, _ = temp[a_idx].sort(dim = -1)
    quantiles = []
    start = 0
    for label_count in labels.unique(return_counts = True, sorted = True)[1]:
        quantiles.append(
            torch.quantile(sorted_temp[start:start + anchor_count * label_count, -contrast_count * (label_count - 1):].to(torch.float),
            1 - thresh,
            dim = -1)
            # interpolation = 'lower')
            )
        start += anchor_count * label_count
    quantiles = torch.cat(quantiles).detach()[inv_idx]
    # quantiles contains the threshold for each row
    threshold_mask = temp > quantiles.view(-1, 1)
    mask = mask * torch.logical_or(threshold_mask, aug_mask)
    # compute log_prob
    #exp_logits = torch.exp(logits) * logits_mask
    exp_logits = torch.exp(logits)
    if not include_diag:
        exp_logits = exp_logits * logits_mask
    if not same_class_negs: 
        exp_logits_mask = 1 - torch.eq(anchor_labels.view(-1,1), contrast_labels.view(1,-1)).cuda(non_blocking=True).detach() # no negatives that are in the same class
        exp_logits = exp_logits * exp_logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

def infoNCE_supcon_all(nn, p, labels, temperature=0.2, gather_all=True):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    if gather_all:
        nn = gather_from_all(nn)
        p = gather_from_all(p)
        labels = gather_from_all(labels)
    n = p.shape[0]
    labels = labels.view(-1, 1)
    labels_cat = labels.repeat(2,1)

    p = torch.cat([nn,p], dim=0)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]

    # mask for positive
    pos_mask = torch.eq(labels, labels_cat.transpose(1,0)).float().cuda(non_blocking=True)
    pos_mask.fill_diagonal_(0.0)
    # mask for positive (only other view) + negatives
    # pos_neg_mask = (~torch.eq(labels, labels.transpose(1,0))).float().cuda(non_blocking=True) + torch.eye(labels.shape[0]).float().cuda(non_blocking=True)
    # denom_exp_logits = torch.exp(logits) * pos_neg_mask
    denom_exp_logits = torch.exp(logits)

    log_prob = -1.0 * (logits - torch.log(denom_exp_logits.sum(dim=1, keepdim=True)))
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(dim=1)

    return mean_log_prob_pos.mean()

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1

class Transform:
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.transform_rotation = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(args.scale[0], args.scale[1])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        y3 = self.transform_rotation(x)
        return y1, y2, y3

class Single_Transform:
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)

        return y1

# rotation
def rotate_images(images, gpu, single=False):
    nimages = images.shape[0]

    if single:
        y = []
        for i in range(nimages):
            y.append(random.randint(0, 3))
            images[i] = torch.rot90(images[i], y[-1], [1, 2])
        y = torch.LongTensor(y).cuda()
        return images.cuda(gpu), y

    n_rot_images = 4 * nimages
    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu,
                                                                                                         non_blocking=True)
    rot_classes = torch.zeros([n_rot_images]).long().cuda(gpu, non_blocking=True)

    rotated_images[:nimages] = images
    # rotate 90
    rotated_images[nimages:2 * nimages] = images.flip(3).transpose(2, 3)
    rot_classes[nimages:2 * nimages] = 1
    # rotate 180
    rotated_images[2 * nimages:3 * nimages] = images.flip(3).flip(2)
    rot_classes[2 * nimages:3 * nimages] = 2
    # rotate 270
    rotated_images[3 * nimages:4 * nimages] = images.transpose(2, 3).flip(3)
    rot_classes[3 * nimages:4 * nimages] = 3

    return rotated_images, rot_classes

def main(args):
    print("Starting Non-DDP training..")
    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp
    
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    main_worker(0, args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RotNet Training')
    parser.add_argument('--data', type=Path, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar100'],
                        help='dataset (imagenet, cifar100)')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
                        help='base learning rate')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--print-freq', default=5, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--save-freq', default=5, type=int, metavar='N',
                        help='save frequency')
    parser.add_argument('--topk-path', type=str, default='./imagenet_resnet50_top10.pkl',
                        help='path to topk predictions from pre-trained classifier')
    parser.add_argument('--checkpoint-dir', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--log-dir', type=Path,
                        metavar='LOGDIR', help='path to tensorboard log directory')
    parser.add_argument('--rotation', default=0.0, type=float,
                        help="coefficient of rotation loss")
    parser.add_argument('--scale', default='0.05,0.14', type=str)
    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    # Training / loss specific parameters
    parser.add_argument('--temp', default=0.2, type=float,
                        help='Temperature for InfoNCE loss')
    parser.add_argument('--mask-mode', type=str, default='',
                        help='Masking mode (masking out only positives, masking out all others than the topk classes',
                        choices=['pos', 'supcon', 'supcon_all', 'topk', 'topk_sum', 'topk_agg_sum', 'weight_anchor_logits', 'weight_class_logits'])
    parser.add_argument('--topk', default=5, type=int, metavar='K',
                        help='Top k classes to use')
    parser.add_argument('--topk-only-first', action='store_true', default=False,
                        help='Whether to only use the first block of anchors')
    parser.add_argument('--memory-bank', action='store_true', default=False,
                        help='Whether to use memory bank')
    parser.add_argument('--mem-size', default=100000, type=int,
                        help='Size of memory bank')
    parser.add_argument('--opt-momentum', default=0.9, type=float,
                        help='Momentum for optimizer')
    parser.add_argument('--optimizer', default='lars', type=str,
                        help='Optimizer', choices=['lars', 'sgd'])

    # Transform
    parser.add_argument('--weak-aug', action='store_true', default=False,
                        help='Whether to use augmentation reguarlization (strong & weak augmentation)')

    # Slurm setting
    parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                        help='number of gpus per node')
    parser.add_argument('--nodes', default=5, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument("--timeout", default=360, type=int,
                        help="Duration of the job")
    parser.add_argument("--partition", default="el8", type=str,
                        help="Partition where to submit")

    parser.add_argument("--exp", default="SimCLR", type=str,
                        help="Name of experiment")
    parser.add_argument('--supcon_new', action='store_true', default=False)
    parser.add_argument('--threshold', default=0.0, type=float)

    parser.add_argument('--rank', default=0, type=int) 
    parser.add_argument('--ddp', action='store_true') 
    args = parser.parse_args()
    
    main(args)
    

'''
python \
      main_supcon.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --epochs 100 \
      --batch-size 64 \
      --learning-rate 4.8 \
      --checkpoint-dir ./saved_models/ \
      --log-dir ./logs/ \
      --rotation 0.0 \
      --exp supcon_new_test \
      --supcon_new \
      --threshold 0.0 
      
'''