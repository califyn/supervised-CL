import os
import math
import time
import copy
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from ldcl.tools.device import get_device, t2np

import torchvision
import torchvision.transforms as T

from resnet import resnet18
from utils import knn_monitor, fix_seed

from PIL import Image

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])

device = get_device()

class RestrictedClassDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset, classes=None, class_map=None):
        super().__init__(dataset.root, transform=dataset.transform, target_transform=dataset.target_transform)

        classes_was_none = classes == None
        if classes_was_none:
            classes = list(set(dataset.targets))
        self.classes = classes

        if class_map != None and not classes_was_none: # only map if classes were provided
            classes = [class_map[c] for c in classes]

        # convert dataset.targets to tensor
        tensor_targets = torch.tensor(dataset.targets)
        class_mask = sum(tensor_targets == class_ for class_ in classes).bool()
        self.data = dataset.data[class_mask]
        self.targets = tensor_targets[class_mask]
        
        for target in classes:
            self.targets[self.targets == target] = torch.tensor(classes.index(target))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def dataset_class_mapper(dataset, classes):
    return RestrictedClassDataset(dataset, classes=classes, class_map={
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }) # cifar map

class ContrastiveLearningTransform:
    def __init__(self):
        transforms = [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]

        self.transform = T.Compose(transforms)

    def __call__(self, x):
        out = [single_transform(self.transform(x)), single_transform(self.transform(x))]
        return out

def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def negative_cosine_similarity_loss(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


def info_nce_loss(z1, z2, temperature=0.5):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, thresh=0.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            thresh: scalar in range 0 to 1, for what fraction of the class should be
                considered as positive pairs
        Returns:
            A loss scalar.
        """

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
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        contrast_labels = labels.repeat(contrast_count).contiguous()
        anchor_labels = labels.repeat(anchor_count).contiguous()

        _, a_idx = torch.sort(anchor_labels, stable = True)
        _, inv_idx = torch.sort(a_idx, stable = True)

        # compute logits
        # anchor_dot_contrast and logits are sorted by label along dimension 0
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # num_labels = torch.unique(labels).max() + 1
        
        # calculating percentiles for positive pairs
        # label_percentile_dists = torch.zeros(num_labels).detach()

        # masks for values in the same class
        mask = torch.eq(anchor_labels.view(-1,1), contrast_labels.view(1,-1))

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask # selects elements of the same class and not along diagonal

        # offset version of anchor_dot_contrast that masks diagonal entries (self) and not in same class
        temp = ((anchor_dot_contrast + 2/self.temperature) * mask)
        
        sorted_temp, _ = temp[a_idx].sort(dim = -1)

        quantiles = []
        start = 0

        for label_count in labels.unique(return_counts = True, sorted = True)[1]:
            quantiles.append(
                torch.quantile(sorted_temp[start:start + anchor_count * label_count, -(contrast_count * label_count + 1):],
                1 - thresh,
                dim = -1,
                interpolation = 'higher')
                )
            start += anchor_count * label_count

        quantiles = torch.cat(quantiles).detach()[inv_idx]

        # quantiles contains the threshold for each row
        threshold_mask = temp >= quantiles.view(-1, 1)

        mask = mask * threshold_mask
            

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_dim, bias=False),
                                 nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Branch(nn.Module):
    def __init__(self, args, encoder=None):
        super().__init__()
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = resnet18()
        self.projector = ProjectionMLP(512, dim_proj[0], dim_proj[1])
        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )
        if args.loss == 'simclr':
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2048, 4))  # output layer
        else:
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.Linear(2048, 4))  # output layer

    def forward(self, x):
        return self.net(x)


def knn_loop(encoder, train_loader, test_loader):
    accuracy = knn_monitor(net=encoder,
                           memory_data_loader=train_loader,
                           test_data_loader=test_loader,
                           k=200 if device != torch.device('mps') else 16, # mps needs a small k
                           device=device,
                           hide_progress=True)
    return accuracy


def ssl_loop(args, encoder=None):
    if args.checkpoint_path:
        print('checkpoint provided => moving to evaluation')
        main_branch = Branch(args, encoder=encoder)
        main_branch.to(device)

        saved_dict = torch.load(os.path.join(args.checkpoint_path))['state_dict']
        main_branch.load_state_dict(saved_dict)
        file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'a')
        file_to_update.write(f'evaluating {args.checkpoint_path}\n')
        return main_branch.encoder, file_to_update

    # logging
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'train_and_eval.log'), 'w')

    # dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=True, transform=ContrastiveLearningTransform() if args.transforms else single_transform, download=True
        ), args.classes),
        shuffle=True,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=True, transform=single_transform, download=True
        ), args.classes),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=False, transform=single_transform, download=True,
        ), args.classes),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # models

    main_branch = Branch(args, encoder=encoder)
    main_branch.to(device)

    if args.loss == 'simsiam':
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        predictor = PredictionMLP(dim_proj[1], args.dim_pred, dim_proj[1])
        predictor.to(device)

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    if args.loss == 'simsiam':
        pred_optimizer = torch.optim.SGD(
            predictor.parameters(),
            momentum=0.9,
            lr=args.lr * args.bsz / 256,
            weight_decay=args.wd
        )

    # macros
    backbone = main_branch.encoder
    projector = main_branch.projector

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))
    scaler = GradScaler()

    # training
    loss_inst = SupConLoss(temperature=args.temperature, base_temperature=args.temperature)
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.loss == 'simsiam':
            predictor.train()

        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            print(f"Step {it % len(train_loader) + 1}/{len(train_loader)}...", end="\r")
            # adjust
            lr = adjust_learning_rate(epochs=args.epochs,
                                      warmup_epochs=args.warmup_epochs,
                                      base_lr=args.lr * args.bsz / 256,
                                      optimizer=optimizer,
                                      loader=train_loader,
                                      step=it)
            # zero grad
            main_branch.zero_grad()
            if args.loss == 'simsiam':
                predictor.zero_grad()

            def forward_step():
                if args.transforms:
                    x1 = inputs[0].to(device)
                    x2 = inputs[1].to(device)
                    b1 = backbone(x1)
                    b2 = backbone(x2)
                    z1 = projector(b1)
                    z2 = projector(b2)
                    z = torch.stack((z1, z2), axis=1)
                else:
                    x = inputs.to(device)
                    b = backbone(x)
                    z = torch.unsqueeze(projector(b), 1)

                # forward pass
                if args.loss == 'simclr':
                    loss = loss_inst(z, labels=y)
                else:
                    raise

                return loss

            # optimization step
            if args.fp16:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_step()
                loss.backward()
                optimizer.step()

        if args.fp16:
            with autocast():
                knn_acc = knn_loop(backbone, memory_loader, test_loader)
        else:
            knn_acc = knn_loop(backbone, memory_loader, test_loader)

        line_to_print = (
            f'epoch: {e} | knn_acc: {knn_acc:.3f} | '
            f'loss: {loss.item():.3f} | lr: {lr:.6f} | '
            f'time_elapsed: {time.time() - start:.3f}'
        )
        if file_to_update:
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
        print(line_to_print)

        if e % args.save_every == 0:
            torch.save(dict(epoch=e, state_dict=main_branch.state_dict()),
                       os.path.join(args.path_dir, f'{e}.pth'))

    return main_branch.encoder, file_to_update


def eval_loop(encoder, file_to_update, ind=None):
    # dataset
    train_transform = T.Compose([
        T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    test_transform = T.Compose([
        T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(32),
        T.ToTensor(),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_class_mapper(torchvision.datasets.CIFAR10('../data', train=True, transform=train_transform, download=True), args.classes),
        shuffle=True,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_class_mapper(torchvision.datasets.CIFAR10('../data', train=False, transform=test_transform, download=True), args.classes),
        shuffle=False,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers
    )

    classifier = nn.Linear(512, 10)
    classifier.to(device)

    # optimization
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=0.9,
        lr=30,
        weight_decay=0
    )
    scaler = GradScaler()

    # training
    for e in range(1, 101):
        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=100,
                                 warmup_epochs=0,
                                 base_lr=30,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                with torch.no_grad():
                    b = encoder(inputs.to(device))
                logits = classifier(b)
                loss = F.cross_entropy(logits, y.to(device))
                return loss

            # optimization step
            if args.fp16:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_step()
                loss.backward()
                optimizer.step()

        if e % 10 == 0:
            accs = []
            classifier.eval()
            for idx, (images, labels) in enumerate(test_loader):
                with torch.no_grad():
                    if args.fp16:
                        with autocast():
                            b = encoder(images.to(device))
                            preds = classifier(b).argmax(dim=1)
                    else:
                        b = encoder(images.to(device))
                        preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.to(device)).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs) * 100
            # final report of the accuracy
            line_to_print = (
                f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
            )
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
            print(line_to_print)

    return accuracy


def main(args):
    fix_seed(args.seed)
    encoder, file_to_update = ssl_loop(args)
    accs = []
    for i in range(5):
        accs.append(eval_loop(copy.deepcopy(encoder), file_to_update, i))
    line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
    file_to_update.write(line_to_print + '\n')
    file_to_update.flush()
    print(line_to_print)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='2048,2048', type=str)
    parser.add_argument('--dim_pred', default=512, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--loss', default='simclr', type=str, choices=['simclr', 'simsiam'])
    parser.add_argument('--save_every', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--path_dir', default='../experiment', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--temperature', default=0.5, type=float)
    
    # Specific to supervised mode
    parser.add_argument('--transforms', action='store_true')
    parser.add_argument('--classes', default=None, type=str)

    args = parser.parse_args()
    device = get_device(idx=args.device)
    if args.classes != None:
        args.classes = args.classes.split(",") # comma-sep

    main(args)
