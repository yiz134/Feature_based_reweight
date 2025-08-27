import jax
import jax.numpy as jnp
from jax import random, lax

import numpy as np
from typing import Tuple

from tensorflow.keras.datasets import cifar10

def load_cifar10(subsample: Tuple[int, int] = (40000, 10000)):
    """
    Load CIFAR-10, normalize using per-channel mean/std, then split into
      - train : first subsample[0] examples of the original train set
      - val   : next   subsample[1] examples of the original train set
      - test  : entire original test set

    Returns:
      X_tr, y_tr, X_val, y_val, X_te, y_te
      where X_* is float32 normalized to zero mean/unit variance per channel,
      shape [N,32,32,3], and y_* are int labels in [0..9].
    """
    (X_tr_full, y_tr_full), (X_te, y_te) = cifar10.load_data()

    # convert to float32 in [0,1]
    X_tr_full = X_tr_full.astype(jnp.float32) / 255.0
    X_te      = X_te.astype(jnp.float32)      / 255.0

    perm = np.random.permutation(len(X_tr_full))
    X_tr_full = X_tr_full[perm]
    y_tr_full = y_tr_full[perm]

    # CIFAR-10 per-channel mean and std (computed over train set)
    cifar10_mean = jnp.array([0.4914, 0.4822, 0.4465], dtype=jnp.float32)
    cifar10_std  = jnp.array([0.2470, 0.2435, 0.2616], dtype=jnp.float32)

    # normalize train+val and test
    X_tr_full = (X_tr_full - cifar10_mean) / cifar10_std
    X_te      = (X_te      - cifar10_mean) / cifar10_std

    n_train, n_val = subsample
    X_tr = X_tr_full[:n_train]
    y_tr = y_tr_full[:n_train].flatten()

    X_val = X_tr_full[n_train : n_train + n_val]
    y_val = y_tr_full[n_train : n_train + n_val].flatten()

    return X_tr, y_tr, X_val, y_val, X_te, y_te.flatten()

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

def create_noisy_cifar(key,
                       subsample=(50000, 0),
                       noise_rate=0.0):
    """
    Returns:
      X_tr, y_tr_noisy,   # subsampled train set with label noise
      X_val, y_val,       # held‐out validation (clean)
      X_te, y_te          # test set (clean)
    """
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_cifar10(subsample)
    y_tr_noisy = y_tr.copy()
    if noise_rate > 0.0:
        y_tr_noisy = inject_label_noise(key, y_tr, noise_rate,
                                  num_classes=10)

    return X_tr, y_tr, y_tr_noisy, X_val, y_val, X_te, y_te

@jax.jit
def augment_batch(
    rng,
    X_batch
):
    """
    Given X_batch of shape (B, 32, 32, 3), returns (X_aug, new_rng) where:
      1) Each image is padded to 40×40 via reflection
      2) Each image is randomly cropped back to 32×32 at a dynamic offset
      3) Each image is randomly flipped horizontally with p=0.5

    All operations use JAX primitives (lax.dynamic_slice, vmap), so this can be jitted.
    """

    B, H, W, C = X_batch.shape
    assert (H, W, C) == (32, 32, 3)

    # 1) Pad to (B, 40, 40, 3) via reflection
    X_padded = jnp.pad(
        X_batch,
        ((0, 0), (4, 4), (4, 4), (0, 0)),
        mode='reflect'
    )

    # 2) Random 32×32 crop from 40×40 padded image
    #    We need dynamic offsets oy, ox in [0..8] for each of the B images
    rng, subkey = jax.random.split(rng)
    offset_y = jax.random.randint(subkey, (B,), 0, 9)  # shape (B,)
    rng, subkey = jax.random.split(rng)
    offset_x = jax.random.randint(subkey, (B,), 0, 9)  # shape (B,)

    # Define a function that uses lax.dynamic_slice to crop one image
    def _crop_one(xi, oy, ox):
        # xi has shape (40, 40, 3)
        # We want a slice of size (32, 32, 3) starting at (oy, ox, 0)
        return lax.dynamic_slice(
            xi,
            start_indices=(oy, ox, 0),   # row=oy, col=ox, channel=0
            slice_sizes=(32, 32, 3)
        )

    # Vectorize cropping over the batch dimension
    cropped = jax.vmap(_crop_one)(X_padded, offset_y, offset_x)
    # cropped has shape (B, 32, 32, 3)

    # 3) Random horizontal flip with probability 0.5
    rng, subkey = jax.random.split(rng)
    flip_probs = jax.random.uniform(subkey, (B,))  # uniform in [0,1)
    do_flip = flip_probs < 0.5                     # boolean mask (B,)

    def _maybe_flip(xi, flip_flag):
        # xi: (32, 32, 3), flip_flag: scalar bool
        # If flip_flag, reverse the width dimension; else leave as is
        return lax.cond(flip_flag, lambda t: t[:, ::-1, :], lambda t: t, xi)

    X_aug = jax.vmap(_maybe_flip)(cropped, do_flip)
    # X_aug has shape (B, 32, 32, 3)

    return X_aug, rng