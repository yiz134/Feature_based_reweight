import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
import numpy as np

import random
import torch

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_rampup(current, rampup_length):
    """Cosine rampup"""
    current = np.clip(current, 0.0, rampup_length)
    return float(-.5 * (np.cos(np.pi * current / rampup_length) - 1))

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def wandbRunlist(config, parse):
    dataset_name = config['name'].split('_')[0]
    lr_scheduler_name = config['lr_scheduler']['type']
    loss_fn_name = config['train_loss']['type']
    
    wandb_run_name_list = []
    
    if parse.distillation:
        if parse.distill_mode == 'eigen':
            wandb_run_name_list.append('distil')
        elif parse.distill_mode == 'fulleigen':
            wandb_run_name_list.append('fulldistill')
        else:
            wandb_run_name_list.append('kmeans')
    else:
        wandb_run_name_list.append('baseline')
    wandb_run_name_list.append(dataset_name)
    wandb_run_name_list.append(lr_scheduler_name)
    wandb_run_name_list.append(loss_fn_name)
    wandb_run_name_list.append(str(config['trainer']['asym']))
    wandb_run_name_list.append(str(config['trainer']['percent']))
    wandb_run_name = '_'.join(wandb_run_name_list)
    
    return wandb_run_name_list

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def plot_weights_hist(weight,
                      train_clean_idx,
                      train_noise_idx,
                      val_idx,
                      bins: int = 50):
    """
    Plot histogram outlines (not normalized) of sample weights for:
      - clean training samples
      - noisy training samples
      - validation samples

    Args:
        weight (array-like or torch.Tensor): 1D array of sample weights, length = total_samples.
        train_clean_idx (array-like or torch.Tensor): indices of clean training samples.
        train_noise_idx (array-like or torch.Tensor): indices of noisy training samples.
        val_idx (array-like or torch.Tensor): indices of validation samples.
        bins (int): Number of bins for the histogram (default: 50).
    """
    # --- convert inputs to numpy ---
    if isinstance(weight, torch.Tensor):
        w = weight.detach().cpu().numpy()
    else:
        w = np.array(weight)

    def to_numpy(idxs):
        if isinstance(idxs, torch.Tensor):
            return idxs.detach().cpu().numpy()
        else:
            return np.array(idxs)

    clean_idx = to_numpy(train_clean_idx)
    noisy_idx = to_numpy(train_noise_idx)
    val_idx = to_numpy(val_idx)

    # --- slice weights by category ---
    w_clean = w[clean_idx]
    w_noisy = w[noisy_idx]
    w_val = w[val_idx]

    # --- plot outlines ---
    plt.figure(figsize=(10, 6))
    plt.hist(
        w_clean, bins=bins, density=False, histtype='step',
        linewidth=2, label='Train clean', color='blue'
    )
    plt.hist(
        w_noisy, bins=bins, density=False, histtype='step',
        linewidth=2, label='Train noisy', color='red'
    )
    plt.hist(
        w_val, bins=bins, density=False, histtype='step',
        linewidth=2, label='Validation', color='green'
    )

    plt.title("Weight Distribution (Counts)\nClean vs. Noisy vs. Validation")
    plt.xlabel("Sample Weight")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weight_dynamic(weight_list,
                        train_clean_idx,
                        train_noise_idx,
                        val_idx,
                        plot_nums=None):
    """
    Plot trajectories of sample weights over time for three groups:
      - clean training samples
      - noisy training samples
      - validation samples

    Args:
        weight_list (list of array-like or torch.Tensor):
            List of weight vectors (length = time steps), each of shape [n_samples].
        train_clean_idx (array-like or torch.Tensor):
            Indices of clean training samples.
        train_noise_idx (array-like or torch.Tensor):
            Indices of noisy training samples.
        val_idx (array-like or torch.Tensor):
            Indices of validation samples.
        plot_nums (dict, optional):
            Number of trajectories to plot per group, keys:
            'train_clean', 'train_noise', 'val'.
            Defaults to plotting all available.
    """
    # -- build a (T, N) numpy array of weights --
    weights_array = np.stack([
        w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else np.array(w)
        for w in weight_list
    ], axis=0)  # shape: [time_steps, n_samples]

    # -- default number of plots per group (all) --
    if plot_nums is None:
        plot_nums = {
            'train_clean': 200,
            'train_noise': 150,
            'val': 50,
        }

    # -- helper to get numpy indices --
    def to_np(idxs):
        if isinstance(idxs, torch.Tensor):
            return idxs.detach().cpu().numpy()
        return np.array(idxs)

    clean_idxs = to_np(train_clean_idx)
    noise_idxs = to_np(train_noise_idx)
    val_idxs = to_np(val_idx)

    # -- plotting setup --
    plt.figure(figsize=(10, 6))
    color_map = {
        'train_clean': ('blue', '--', 'Train Clean'),
        'train_noise': ('red', ':', 'Train Noisy'),
        'val': ('green', '-', 'Validation'),
    }
    added = {k: False for k in color_map}

    def plot_group(idxs, group_name):
        num = plot_nums.get(group_name, 0)
        if num <= 0 or len(idxs) == 0:
            return
        color, style, label = color_map[group_name]
        alpha = 0.8 if group_name == 'val' else 0.5
        for i in idxs[:num]:
            plt.plot(
                weights_array[:, i],
                linestyle=style,
                linewidth=1.0,
                color=color,
                alpha=alpha,
                label=label if not added[group_name] else None
            )
            added[group_name] = True

    # -- plot each group --
    plot_group(clean_idxs, 'train_clean')
    plot_group(noise_idxs, 'train_noise')
    plot_group(val_idxs, 'val')

    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Weight Value")
    plt.title("Dynamics of Weight Change")
    plt.tight_layout()
    plt.show()


def plot_value_distributions(Kmat, y_noisy, y_clean, y_val, num_class=10, bins=50, mean=False):
    """
    For each sample i, plot a figure showing the kernel‐value distributions
    Kmat[i, :] grouped by class in y_val.

    - y_clean[i] and y_noisy[i] curves are drawn with solid lines.
    - all other classes are drawn with dashed lines.
    - each class has its own color.
    """
    Kmat = np.asarray(Kmat.cpu())
    y_val = np.asarray(y_val.cpu())
    y_clean = np.asarray(y_clean.cpu())
    y_noisy = np.asarray(y_noisy.cpu())

    n = len(y_clean)
    for i in range(n):
        vals_i = Kmat[i]  # shape (N,)
        true_k = y_clean[i]
        noisy_k = y_noisy[i]
        if mean:
            means = []
            for k in range(num_class):
                mask = (y_val == k)
                if mask.any():
                    means.append(vals_i[mask].mean())
                else:
                    means.append(np.nan)

            plt.figure(figsize=(6, 4))
            bars = plt.bar(
                np.arange(num_class), means,
                color=['C{}'.format(k) for k in range(num_class)],
                alpha=0.7
            )
            bars[true_k].set_edgecolor('k')
            bars[true_k].set_linewidth(2)
            bars[noisy_k].set_hatch('//')

            plt.xlabel('Class k')
            plt.ylabel('Mean kernel value')
            plt.title(f'Sample {i}: class {true_k} (clean), {noisy_k} (noisy)')
            plt.xticks(np.arange(num_class))
            plt.show()
        else:
            plt.figure(figsize=(6, 4))
            for k in range(num_class):
                # select all similarities to points of class k
                mask = (y_val == k)
                if not mask.any():
                    continue

                data_k = vals_i[mask]
                # histogram density estimate
                hist, edges = np.histogram(data_k, bins=bins, range=(-10, 10), density=False)
                centers = (edges[:-1] + edges[1:]) / 2

                # choose line style
                ls = '-' if (k == true_k or k == noisy_k) else '--'
                plt.plot(centers, hist, label=f'class {k}', linestyle=ls)

            plt.xlabel('Similarity value')
            plt.ylabel('Frequency')
            plt.title(f'Sample {i}  —  clean={true_k}, noisy={noisy_k}')
            plt.legend(ncol=2, fontsize='small')
            plt.tight_layout()
            plt.show()


def get_noisy_high_weight_idx(weights, y_noisy, y_clean, idx_type='noisy_positive'):
    noise_mask = (y_noisy.cpu() != y_clean.cpu()) #noise mask
    weight_mask = (weights.cpu() > 0.1) #positive mask
    if idx_type == 'noisy_positive':
        mask = noise_mask * weight_mask
    elif idx_type == 'noisy_negative':
        mask = noise_mask * (~weight_mask)
    elif idx_type == 'clean_positive':
        mask = (~noise_mask) * (weight_mask)
    elif idx_type == 'clean_negative':
        mask = (~noise_mask) * (~weight_mask)
    idxs = mask.nonzero()
    return idxs.squeeze(1)


def transform_Kmat(Kmat, y_val, num_class=10):
    one_hot = F.one_hot(y_val, num_classes=num_class).float()
    sums = Kmat @ one_hot  # [B, C]
    counts = one_hot.sum(dim=0)
    means = sums / counts.unsqueeze(0)  # [B, C]
    top2 = means.topk(k=2, dim=1).values
    max1 = top2[:, 0]
    max2 = top2[:, 1]
    shift = top2[:, 1]
    class_idx = torch.arange(num_class, device=y_val.device, dtype=y_val.dtype)
    return means - shift.unsqueeze(1), class_idx, means


def get_noisy_rate(weight, y_clean, y_noisy, thr=0.0):
    err_mask = (y_clean.cpu() != y_noisy.cpu())
    total_err = err_mask.sum().item()
    if total_err == 0:
        return 0.0
    high_weight_count = (weight[err_mask] > thr).sum().item()
    return high_weight_count/((weight > thr).sum().item())
