import jax
import jax.numpy as jnp
from jax import lax

from load_data.data_utils import *
from torchvision.datasets import CIFAR10
import torch
from typing import Tuple
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def get_noisy_labels(y, noisy_idx, num_classes: int = 10) -> torch.Tensor:
    y_noisy = y.clone()
    for i in noisy_idx:
        orig = y_noisy[i].item()
        choices = torch.tensor([c for c in range(num_classes) if c != orig], device=y.device)
        y_noisy[i] = choices[torch.randint(len(choices), (1,), device=y.device)]
    return y_noisy


class TrainSplitDataset(Dataset):
    def __init__(self,
                 images_np: 'np.ndarray',
                 clean_labels: torch.Tensor,
                 noisy_labels: torch.Tensor,
                 indices: torch.Tensor,
                 transform=None):
        self.images = images_np
        self.clean = clean_labels
        self.noisy = noisy_labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return self.clean.size(0)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])  # H×W×C uint8 -> PIL
        if self.transform:
            img = self.transform(img)  # ToTensor + augment + normalize
        y_clean = self.clean[idx].item()
        y_noise = self.noisy[idx].item()
        idx = self.indices[idx].item()
        return img, y_clean, y_noise, idx


def load_cifar(subsample: Tuple[int, int] = (40000, 10000),
               noise_rate: float = 0.0,
               device: str = 'cpu',
               batch_size: int = 128,
               num_workers: int = 0):
    """
    Return:
      train_loader: (img, y_clean, y_noise, idx)
      test_loader:  (img, y)
      X_val, y_val: tensors on `device`
    """
    # 0) Transforms
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2470, 0.2435, 0.2616])
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 1) Load raw arrays (no transform)
    raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    images_np = raw.data  # numpy (50000,32,32,3)
    labels_all = torch.tensor(raw.targets)  # [50000]
    perm = torch.randperm(len(labels_all))
    images_np = images_np[perm.numpy()]
    labels_all = labels_all[perm.numpy()]

    # 2) Shuffle indices before split
    n_train, n_val = subsample
    assert n_train + n_val <= len(labels_all)
    perm = torch.randperm(len(labels_all))
    train_idx = perm[:n_train]
    num_noisy = int(noise_rate * len(train_idx))
    shuffled_train = train_idx[torch.randperm(len(train_idx))]

    train_noise_idx = shuffled_train[:num_noisy]
    train_clean_idx = shuffled_train[num_noisy:]
    val_idx = perm[n_train:n_train + n_val]

    train_images = images_np
    val_images = images_np[val_idx.numpy()]

    y_noisy_train = get_noisy_labels(labels_all, train_noise_idx, num_classes=10).to(device)
    y_clean_train = labels_all.to(device)
    y_val = labels_all[val_idx].to(device)

    # 5) Build train_loader
    train_ds = TrainSplitDataset(
        images_np=train_images,
        clean_labels=y_clean_train,
        noisy_labels=y_noisy_train,
        indices=torch.arange(0, len(train_images)),
        transform=train_tf
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 6) Build test_loader
    test_ds = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=eval_tf
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 7) Precompute X_val, y_val
    X_val_list = [eval_tf(Image.fromarray(val_images[i])) for i in range(n_val)]
    X_val = torch.stack(X_val_list, dim=0).to(device)  # [n_val,3,32,32]
    y_val = y_val.to(device)  # [n_val]

    return train_loader, test_loader, X_val, y_val, train_clean_idx, train_noise_idx, val_idx


def keep_k_true(mask: torch.Tensor, k: int) -> torch.Tensor:
    assert mask.dtype == torch.bool
    flat = mask.view(-1)
    true_idx = torch.where(flat)[0]
    k = min(int(k), int(true_idx.numel()))
    if k <= 0:
        return torch.zeros_like(mask)
    if k == true_idx.numel():
        return mask.clone()

    perm = torch.randperm(true_idx.numel(), device=mask.device)[:k]
    keep_idx = true_idx[perm]

    out = torch.zeros_like(flat, dtype=torch.bool)
    out[keep_idx] = True
    return out.view_as(mask)

def add_symmetric_noise(labels_all: torch.Tensor,
                        noise_rate: float,
                        num_classes: int = 10) -> torch.Tensor:
    assert labels_all.dtype in (torch.int64, torch.long)
    n = labels_all.numel()
    k = int(noise_rate * n)
    if k <= 0:
        return labels_all.clone()

    y_noisy = labels_all.clone()
    device = labels_all.device

    idx = torch.randperm(n, device=device)[:k]

    r = torch.randint(0, num_classes - 1, (k,), device=device)
    new = r + (r >= y_noisy[idx]).long()

    y_noisy[idx] = new
    return y_noisy

def add_asymmetric_noise():
    pass


def load_cifar_new(subsample: Tuple[int, int] = (40000, 10000),
               noise_rate: float = 0.0,
               device: str = 'cpu',
               batch_size: int = 128,
               num_workers: int = 0,
               noise_type=None):
    """
    Return:
      train_loader: (img, y_clean, y_noise, idx)
      test_loader:  (img, y)
      X_val, y_val: tensors on `device`
    """
    # 0) Transforms
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2470, 0.2435, 0.2616])
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    n_train, n_val = subsample
    # 1) Load raw arrays (no transform)
    raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    images_np = raw.data  # numpy (50000,32,32,3)
    labels_all = torch.tensor(raw.targets)  # [50000]

    ######################################################
    if noise_type == 'symmetric':
        noisy_labels = add_symmetric_noise(labels_all, noise_rate)
    elif noise_type == 'asymmetric':
        noisy_labels = add_asymmetric_noise(labels_all, noise_rate)
    elif noise_type == 'cifar10-aggre':
        noise = torch.load('./data/CIFAR-10_human.pt')
        noisy_labels = torch.tensor(np.array(noise['aggre_label']))
    elif noise_type == 'cifar10-worse':
        noise = torch.load('./data/CIFAR-10_human.pt')
        noisy_labels = torch.tensor(np.array(noise['worse_label']))
    elif noise_type == 'cifar10-random1':
        noise = torch.load('./data/CIFAR-10_human.pt')
        noisy_labels = torch.tensor(np.array(noise['random_label1']))
    elif noise_type == 'cifar10-random2':
        noise = torch.load('./data/CIFAR-10_human.pt')
        noisy_labels = torch.tensor(np.array(noise['random_label2']))
    elif noise_type == 'cifar10-random3':
        noise = torch.load('./data/CIFAR-10_human.pt')
        noisy_labels = torch.tensor(np.array(noise['random_label3']))
    elif noise_type == "inst":
        pass
    else:
        noisy_labels = labels_all


    clean_indicator = (labels_all==noisy_labels)
    val_indicator = keep_k_true(clean_indicator, n_val)

    assert n_train + n_val <= len(labels_all)
    perm = torch.randperm(len(labels_all))
    train_images = images_np[perm.numpy()]
    clean_labels = labels_all[perm].to(device)
    noisy_labels = noisy_labels[perm].to(device)
    clean_indicator = clean_indicator[perm]
    val_indicator = val_indicator[perm]

    val_images = train_images[val_indicator.cpu().numpy()]
    y_val = clean_labels[val_indicator]

    train_clean_idx = torch.where(clean_indicator)[0]
    train_noise_idx = torch.where(~clean_indicator)[0]
    val_idx = torch.where(val_indicator)[0]

    X_val_list = [eval_tf(Image.fromarray(val_images[i])) for i in range(n_val)]
    X_val = torch.stack(X_val_list, dim=0).to(device)  # [n_val,3,32,32]
    y_val = y_val.to(device)

    train_ds = TrainSplitDataset(
        images_np=train_images,
        clean_labels=clean_labels,
        noisy_labels=noisy_labels,
        indices=torch.arange(0, len(train_images)),
        transform=train_tf
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 6) Build test_loader
    test_ds = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=eval_tf
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader, X_val, y_val, train_clean_idx, train_noise_idx, val_idx
