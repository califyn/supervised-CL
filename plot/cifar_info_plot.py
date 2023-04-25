from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA
import argparse

import subprocess

import torch
import torchvision
import torchvision.transforms as T
from ascii_magic import AsciiArt
from PIL import Image

import numpy as np

from tqdm import tqdm
import os
import copy

from ldcl.models.cifar_resnet import resnet18

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
        
        target_copy = copy.deepcopy(self.targets)
        for target in classes:
            self.targets[target_copy == target] = torch.tensor(classes.index(target))
        print(self.targets)

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
    if classes != None:
        classes = classes.split(",")
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
class ProjectionMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_dim, bias=False),
                                 torch.nn.BatchNorm1d(hidden_dim),
                                 torch.nn.ReLU(inplace=True),
                                 torch.nn.Linear(hidden_dim, out_dim, bias=False),
                                 torch.nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)

class Branch(torch.nn.Module):
    def __init__(self, args, encoder=None):
        super().__init__()
        dim_proj = [int(x) for x in args.dim_proj.split(',')]
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = resnet18()
        self.projector = ProjectionMLP(512, dim_proj[0], dim_proj[1])
        self.net = torch.nn.Sequential(
            self.encoder,
            self.projector
        )
        if args.loss == 'simclr':
            self.predictor2 = torch.nn.Sequential(torch.nn.Linear(512, 2048),
                                            torch.nn.LayerNorm(2048),
                                            torch.nn.ReLU(inplace=True),  # first layer
                                            torch.nn.Linear(2048, 2048),
                                            torch.nn.LayerNorm(2048),
                                            torch.nn.ReLU(inplace=True),
                                            torch.nn.Linear(2048, 4))  # output layer
        else:
            self.predictor2 = torch.nn.Sequential(torch.nn.Linear(512, 2048),
                                            torch.nn.LayerNorm(2048),
                                            torch.nn.ReLU(inplace=True),  # first layer
                                            torch.nn.Linear(2048, 2048),
                                            torch.nn.LayerNorm(2048),
                                            torch.nn.Linear(2048, 4))  # output layer

    def forward(self, x):
        return self.net(x)

device = get_device(idx=7)
num_cols_ascii = 32

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
unnormalize = T.Normalize([-2.43, -2.42, -2.22], [4.94, 5.01, 4.98])
single_transform = T.Compose([T.ToTensor(), normalize])

def cifar_reverse_map(i):
    return {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }[i]

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def to_image(imgs):
    imgs = unnormalize(imgs)
    imgs = imgs.cpu().numpy() * 255
    vhex = np.vectorize(lambda x: hex(x)[2:].zfill(2))
    imgs = vhex(imgs.astype(int))
    pxls = np.char.add(np.char.add(imgs[:, 0], imgs[:, 1]), imgs[:, 2])
    pxls = np.char.add(np.char.add(np.full(pxls.shape, "<span style='color:#"), pxls), np.full(pxls.shape, ";'>&#9608;</span>"))
    pxls = pxls.tolist()
    rows = [["".join(pxls[i][j]) for j in range(len(pxls[0]))] for i in range(len(pxls))]
    imgs = ["".join(["<br>" + x for x in rows[i]]) for i in range(len(rows))]
    return imgs

def main_plot(args):
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=not args.test, transform=single_transform, download=True,
        ), args.classes),
        shuffle=False,
        batch_size=512,
        pin_memory=True,
        num_workers=8,
    )
    branch = Branch(args)
    state_dict = torch.load(os.path.join(args.path_dir, f"{args.id}.pth"), map_location=device)["state_dict"]

    branch.load_state_dict(state_dict)
    branch.to(device)
    branch.eval()

    embeds = []
    targets = []
    imgs = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            ind = batch[0].type(torch.float32).to(device)
            embeds.append(branch(ind))
            targets = targets + batch[1].cpu().numpy().tolist()
            if args.preview and len(imgs) <= 1000:
                #rl_imgs = [T.ToPILImage()(unnormalize(batch[0][i])) for i in range(len(batch[0]))]
                #rl_imgs = [AsciiArt.from_pillow_image(img).to_html(columns=num_cols_ascii, width_ratio=1.0, char="&#9608;") for img in rl_imgs] # create ASCII image previews
                #imgs = imgs + ["<br>" + rl_imgs[i] for i in range(len(rl_imgs))]
                imgs = imgs + to_image(batch[0])
            else:
                imgs = imgs + ["" for i in range(len(batch[0]))]
    embeds = torch.cat(embeds, dim=0).detach().cpu().numpy()
    targets = np.array(targets)
    vals = {
        "targets": targets,
        "nl_targets": list(map(cifar_reverse_map, targets)),
        "ids": list(range(len(targets))),
    }
    if args.preview:
        vals["imgs"] = imgs

    if args.normalize:
        embeds = embeds / np.sqrt(np.sum(np.square(embeds), axis=1))[:, np.newaxis]
    """
    means = []
    for i in range(10):
        means.append(np.mean(embeds[vals["targets"] == i], axis=0))
    means = np.array(means)
    print(means)
    """

    """
    mask = np.equal(vals["targets"], 2)
    class_embeds = embeds[mask]
    class_vals = {}
    for key in vals:
        class_vals[key] = vals[key][mask]
    """

    """
    # Dim reduction (2d only).
    pca = PCA(n_components=2) # dimensionality reduction for 2D
    single_orbit_embeds = pca.fit_transform(single_orbit_embeds)
    oneD_span_embeds = pca.transform(oneD_span_embeds)
    """

    # Plot

    def cmap_three():
        nonlocal embeds

        plot = VisPlot(3, num_subplots=5) # 3D plot, 2 for 2D plot
        print(embeds.shape)
        plot.add_with_cmap(embeds, vals, cmap=["husl", "viridis", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "x", "v.x"], size=1.5, outline=False)
        plot.add_with_cmap(so_embeds, so_vals, cmap=["husl", "viridis", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "x", "v.x"], size=2.5, outline=True)

        return plot

    def cmap_one():
        plot = VisPlot(3)
        plot.add_with_cmap(embeds, vals, cmap="tab10", cby="targets", size=1.5, outline=False)
        #plot.add_with_cmap(class_embeds, class_vals, cmap="tab10", cby="targets", size=3, outline=True)

        return plot

    plot = cmap_one()

    plot.show()
    if args.server:
        subprocess.run('python -m http.server', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--croplow', type=float)
    #parser.add_argument('--epochs', type=int)
    parser.add_argument('--path_dir', type=str)
    parser.add_argument('--id', default='final', type=str)
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--dim_proj', default="2048,3", type=str)
    parser.add_argument('--loss', default="simclr", type=str)
    parser.add_argument('--classes', default=None, type=str)
    parser.add_argument('--normalize', action='store_true')

    args = parser.parse_args()
    main_plot(args)
