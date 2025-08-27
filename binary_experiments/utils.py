import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def hist_init_preds(output):
    save_name = 'plots/init_res_hist.png'
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(output.detach().cpu().numpy(), bins=50)
    plt.title('Initial Predictions Distribution')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    plt.savefig(save_name)
    plt.close()



def one_hot_to_signed_labels(one_hot_tensor):
    """
    Convert a binary 2D one-hot tensor to a label tensor with values in {-1, 1}.

    Parameters:
        one_hot_tensor (torch.Tensor): A [n_samples, 2] tensor with one-hot encoded binary labels.

    Returns:
        torch.Tensor: A [n_samples] tensor with values -1 or 1.
    """
    if one_hot_tensor.shape[1] != 2:
        raise ValueError("This function is only for binary classification with 2 classes.")

    if not torch.all((one_hot_tensor == 0) | (one_hot_tensor == 1)):
        raise ValueError("Input must be a one-hot encoded tensor with 0s and 1s.")

    if not torch.all(one_hot_tensor.sum(dim=1) == 1):
        raise ValueError("Each row must have exactly one '1' for valid one-hot encoding.")

    # Convert one-hot [0, 1] → 1, [1, 0] → -1
    return one_hot_tensor[:, 1] * 2 - 1


def mean_center_K(KXY):
    row_mean = KXY.mean(dim=1, keepdim=True)       # shape (n, 1)
    col_mean = KXY.mean(dim=0, keepdim=True)       # shape (1, m)
    total_mean = KXY.mean()                        # scalar

    K_centered = KXY - row_mean - col_mean + total_mean
    return K_centered



def distribution_K(KXY: torch.Tensor, ind_r: int = 500, ind_c: int = 100):
    A = KXY[:ind_r, :ind_c].flatten().cpu().numpy()
    B = KXY[:ind_r, ind_c:].flatten().cpu().numpy()
    C = KXY[ind_r:, :ind_c].flatten().cpu().numpy()
    D = KXY[ind_r:, ind_c:].flatten().cpu().numpy()

    all_vals = np.concatenate([A, B, C, D])
    min_val, max_val = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(min_val, max_val, 100)

    # 绘制四条密度曲线（normalized histogram）
    plt.figure(figsize=(10, 6))
    plt.hist(A, bins=bins, density=True, histtype='step', linewidth=2, label='Top-Left (A)', color='blue')
    plt.hist(B, bins=bins, density=True, histtype='step', linewidth=2, label='Top-Right (B)', color='green')
    plt.hist(C, bins=bins, density=True, histtype='step', linewidth=2, label='Bottom-Left (C)', color='orange')
    plt.hist(D, bins=bins, density=True, histtype='step', linewidth=2, label='Bottom-Right (D)', color='red')

    plt.title("Distribution of Entries in Four Kernel Quadrants")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()