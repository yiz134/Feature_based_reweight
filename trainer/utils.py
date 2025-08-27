import torch
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch.nn.functional as F
def make_lr_schedule(m: int, batch_size: int, num_epochs: int, base_lr: float):
    steps_per_epoch = math.ceil(m / batch_size)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    def lr_lambda(step):
        # Warmup phase
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # Main phase with piecewise constant
        if step >= 250 * steps_per_epoch:
            return 0.01  # 0.1 * 0.1
        elif step >= 150 * steps_per_epoch:
            return 0.1
        else:
            return 1.0

    return lr_lambda

def make_decay_mask(model):
    """
    Create a mask for weight decay.
    In PyTorch, we typically apply weight decay to all parameters except biases and batch normalization parameters.
    """
    decay_mask = []
    for name, param in model.named_parameters():
        # Don't apply weight decay to biases and batch norm parameters
        if 'bias' in name or 'bn' in name:
            decay_mask.append(False)
        else:
            decay_mask.append(True)
    return decay_mask

def get_optimizer_and_scheduler(model, m, batch_size, num_epochs, base_lr):
    # Create optimizer with weight decay
    decay_mask = make_decay_mask(model)
    params = []
    for param, should_decay in zip(model.parameters(), decay_mask):
        if should_decay:
            params.append({'params': param, 'weight_decay': 5e-4})
        else:
            params.append({'params': param, 'weight_decay': 0})
    
    optimizer = torch.optim.SGD(
        params,
        lr=base_lr,
        momentum=0.9
    )
    
    # Create learning rate scheduler
    lr_lambda = make_lr_schedule(m, batch_size, num_epochs, base_lr)
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def make_decay_mask(params):
    """
    Recursively traverse `params` (which is a FrozenDict). For every leaf
    (i.e. array), return True if its key-name is "kernel", else False.
    Return a new FrozenDict of booleans with identical tree structure.
    """
    def recurse(tree):
        if isinstance(tree, FrozenDict):
            # Recurse into each sub‐FrozenDict
            return FrozenDict({k: recurse(v) for k, v in tree.items()})
        else:
            # At a leaf: tree is an ndarray. The caller's key told us whether
            # this leaf's name was "kernel"—but here we only reach the leaf value,
            # so instead we rely on the fact that in recurse's caller we know the key.
            # To work around that, we'll rewrite this function to accept (tree, key).
            raise RuntimeError("Should not hit leaf without key context")
    # Instead, define a helper that carries the key in recursion:
    def recurse_with_key(tree, key_name):
        if isinstance(tree, FrozenDict):
            return FrozenDict({k: recurse_with_key(v, k) for k, v in tree.items()})
        else:
            # leaf array; use the most‐recent key_name
            return (key_name == "kernel")
    return recurse_with_key(params, None)


def compute_pi_inv(y_tr: torch.Tensor):
    """
    Compute inverse class frequency weights for each sample.

    Args:
        y_tr (torch.Tensor): Tensor of shape (n,) with integer class labels.

    Returns:
        torch.Tensor: Tensor of shape (n,), where each value is n / count[class],
                      representing the inverse class frequency weight for each sample.
    """
    n = y_tr.shape[0]
    counts = torch.bincount(y_tr)
    weights = n / counts[y_tr]
    return weights


def weighted_mean_features(features, y):
    pi_inv = compute_pi_inv(y) #shape (n, )
    pi_inv = pi_inv / pi_inv.sum() 
    return pi_inv.unsqueeze(0) @ features 

def update_weight(weight, w_direction, ratio=None, clipmax=2, eta=None):
    """
    Compute updated weight by weight + eta * w_direction and clip with 0 
    and choose eta so that ratio * len(weight) entries become 0.
    If ratio=None or ratio*len(w)<# of negative entries in w_direction, 
    update weight so that all entries where w_direction<0 becomes 0. 
    """
    if (w_direction >= 0).all().item():
        weight.fill_(clipmax)
        return weight
    if eta is None:
        n = weight.numel()

        neg_mask = w_direction < 0
        t = -weight[neg_mask] / w_direction[neg_mask]  # shape (M,), M = #neg

        M = t.numel()
        if M == 0:
            return weight.clone()

        if ratio is None or ratio * n > M:
            k = M
        else:
            k = int(ratio * n)

        if k == 0:
            # If k==0，η=0 → do not update
            return weight.clone()

        t_sorted, _ = torch.sort(t)
        eta = t_sorted[k - 1]
        w_new = weight + eta * w_direction
    else:
        w_new = weight + eta * w_direction
    w_new = torch.clamp(w_new, min=0.0)
    if clipmax:
        w_new = torch.clamp(w_new, max=clipmax)
    return w_new

def label_correction(Kmat: torch.Tensor,
                     y_tr: torch.Tensor,
                     y_val: torch.Tensor,
                     w_direction: torch.Tensor,
                     correction_rate: float = 1.0) -> torch.Tensor:
    N_tr = y_tr.size(0)
    n_class = int(y_val.max().item()) + 1
    if correction_rate < 1:
        base_mask = w_direction < 0
        idxs = base_mask.nonzero(as_tuple=False).view(-1)
        M = idxs.numel()
        num_to_correct = int(correction_rate * M)
        if num_to_correct > 0:
            sorted_idxs = idxs[torch.argsort(w_direction[idxs])]
            selected = sorted_idxs[:num_to_correct]
            neg_mask = torch.zeros_like(base_mask)
            neg_mask[selected] = True
        else:
            neg_mask = torch.zeros_like(base_mask)
    else:
        neg_mask = w_direction < 0

    if neg_mask.sum().item() == 0:
        return y_tr.clone()

    one_hot = F.one_hot(y_val, num_classes=n_class).to(Kmat.dtype)  # (N_val, n_class)

    class_sums = Kmat[neg_mask].matmul(one_hot)  # (N_tr_sub, n_class)

    max_vals, max_cls = class_sums.max(dim=1)  # (N_tr_sub,), (N_tr_sub,)

    y_tr_new = y_tr.clone()
    y_tr_new[neg_mask] = max_cls
    return y_tr_new

def visualize_Kmat(Kmat, y):
    K_np = Kmat.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    c = int(y_np.max()) + 1

    Mat = np.zeros((c, c), dtype=K_np.dtype)

    for i in range(c):
        idx_i = np.where(y_np == i)[0]
        for j in range(c):
            idx_j = np.where(y_np == j)[0]
            if idx_i.size and idx_j.size:
                Mat[i, j] = K_np[np.ix_(idx_i, idx_j)].mean()
            else:
                Mat[i, j] = np.nan

    plt.figure(figsize=(6, 5))
    im = plt.imshow(Mat, interpolation='nearest')
    plt.colorbar(im)
    plt.xlabel('Class j')
    plt.ylabel('Class i')
    plt.title('Class–Class Mean Kernel Matrix')
    plt.show()


def get_K_val_class_wise(val_features, y_val):
    K_val = val_features @ val_features.T
    K_np = K_val.detach().cpu().numpy()
    y_np = y_val.detach().cpu().numpy()
    c = int(y_np.max()) + 1

    Mat = np.zeros((c, c), dtype=K_np.dtype)
    for i in range(c):
        idx_i = np.where(y_np == i)[0]
        for j in range(c):
            idx_j = np.where(y_np == j)[0]
            if idx_i.size and idx_j.size:
                Mat[i, j] = K_np[np.ix_(idx_i, idx_j)].mean()
            else:
                Mat[i, j] = np.nan

    return torch.tensor(Mat).to(val_features.device)



def save_checkpoint(model, optimizer, epoch, loss, lr_scheduler=None, path="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, lr_scheduler=None, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 继续下一个epoch
    loss = checkpoint['loss']
    return model, optimizer, lr_scheduler, start_epoch, loss