from jax import random
import numpy as np
import torch

def inject_label_noise(key, y, noise_rate, num_classes):
    y_noisy = y.copy()
    n = y_noisy.shape[0]
    num_noisy = int(noise_rate * n)
    noisy_indices = random.choice(key, n, shape=(num_noisy,), replace=False)
    for idx in noisy_indices:
        orig = int(y_noisy[idx])
        choices = list(range(num_classes))
        y_noisy[idx] = np.random.choice(choices)
    return y_noisy



def to_tensor(arrays, device=None, dtype=None):
    """
    Convert each element in a list of JAX arrays or NumPy arrays to a PyTorch tensor.

    Args:
        arrays (list): List of arrays, each can be a JAX Array, NumPy ndarray, or array-like.
        device (torch.device or str, optional): Device to move each tensor to. Defaults to None.
        dtype (torch.dtype, optional): Desired data type for the tensors. Defaults to None.

    Returns:
        list[torch.Tensor]: A list of tensors corresponding to each input array.
    """
    try:
        import jax
        from jax import device_get
        _has_jax = True
    except ImportError:
        _has_jax = False
    tensors = []
    for x in arrays:
        # If JAX is available and this is a JAX array, move to host as NumPy
        if _has_jax and isinstance(x, jax.Array):
            x = device_get(x)
        # Convert to NumPy array
        arr_np = np.array(x)
        # Convert NumPy to PyTorch tensor
        tensor = torch.from_numpy(arr_np)
        # Cast to desired dtype
        if dtype is not None:
            tensor = tensor.to(dtype)
        # Move to device if specified
        if device is not None:
            tensor = tensor.to(device)
        tensors.append(tensor)
    return tensors


def get_noisy_labels(y, noise_rate=0.0, num_classes=10):
    y_noisy = y.clone()
    n = y_noisy.shape[0]
    num_noisy = int(noise_rate * n)
    
    # Generate random indices for noisy samples
    noisy_indices = torch.randperm(n, device=y.device)[:num_noisy]
    
    # Generate random labels for noisy samples
    for idx in noisy_indices:
        orig = y_noisy[idx].item()
        choices = torch.tensor([i for i in range(num_classes) if i != orig], device=y.device)
        y_noisy[idx] = choices[torch.randint(0, len(choices), (1,), device=y.device)]
    
    return y_noisy
