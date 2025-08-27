import math
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from tqdm import tqdm

from projection import *
from data import *
from resnet import *

def make_lr_schedule(m: int, batch_size: int, num_epochs: int, base_lr: float):
    steps_per_epoch = math.ceil(m / batch_size)
    total_steps = num_epochs * steps_per_epoch

    # warmup for first 5 epochs
    warmup_steps = 5 * steps_per_epoch
    warmup = optax.linear_schedule(
        init_value=0.0, end_value=base_lr, transition_steps=warmup_steps
    )

    # piecewise after epoch 150 and 250
    boundary_150 = 150 * steps_per_epoch
    boundary_250 = 250 * steps_per_epoch
    main = optax.piecewise_constant_schedule(
        init_value=base_lr,
        boundaries_and_scales={
            boundary_150: 0.1,
            boundary_250: 0.1,
        },
    )

    lr_schedule = optax.join_schedules(
        schedules=[warmup, main],
        boundaries=[warmup_steps],
    )
    return lr_schedule

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

def compute_pi_inv(y_tr: jnp.ndarray):
    n = y_tr.shape[0]
    counts = jnp.bincount(y_tr)
    return n / counts[y_tr]  # shape (n,)

@jax.jit
def train_step(state, xb, yb, wb):
    def loss_fn(params, batch_stats):
        vars = {"params": params, "batch_stats": batch_stats}
        (logits, new_vars) = state.apply_fn(vars, xb, train=True, mutable=["batch_stats"])
        y_oh = jax.nn.one_hot(yb, logits.shape[-1])
        ce = -jnp.sum(y_oh * jax.nn.log_softmax(logits), axis=-1)
        loss = jnp.mean(wb * ce)
        return loss, new_vars["batch_stats"]

    (loss, new_bs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.batch_stats
    )
    state = state.apply_gradients(grads=grads, batch_stats=new_bs)
    return state, loss

@jax.jit
def eval_step(state, xb, wb=None, yb=None):
    vars = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(vars, xb, train=False, mutable=False)
    if (wb is not None) and (yb is not None):
        y_oh = jax.nn.one_hot(yb, logits.shape[-1])
        ce = -jnp.sum(y_oh * jax.nn.log_softmax(logits), axis=-1)
        loss = jnp.sum(wb * ce) / jnp.sum(wb)
        return loss, logits
    return logits

@jax.jit
def compute_residuals(apply_fn, params, batch_stats, X, y):
    vars = {"params": params, "batch_stats": batch_stats}
    logits = apply_fn(vars, X, train=False, mutable=False)
    probs = jax.nn.softmax(logits)
    idx = jnp.arange(y.shape[0])
    return probs[idx, y] - 1.0

def evaluate_in_batches(state, X_all, y_all=None, w_all=None, batch_size=512):
    N = X_all.shape[0]
    num_steps = int(jnp.ceil(N / batch_size))

    all_logits = []
    total_num = 0.0
    total_den = 0.0

    for i in range(num_steps):
        start = i * batch_size
        end = min((i + 1) * batch_size, N)
        xb = X_all[start:end]

        if (w_all is None) or (y_all is None):
            logits = eval_step(state, xb)
            all_logits.append(logits)
        else:
            wb = w_all[start:end]
            yb = y_all[start:end]
            loss_chunk, logits = eval_step(state, xb, wb, yb)
            all_logits.append(logits)
            sum_wb = float(jnp.sum(wb))
            total_den += sum_wb
            total_num += float(loss_chunk) * sum_wb

    all_logits = jnp.concatenate(all_logits, axis=0)
    if (w_all is None) or (y_all is None):
        return all_logits
    avg_loss = total_num / total_den
    return avg_loss, all_logits

def reweight_step(
    state,
    X_tr, y_tr, w, X_val, y_val,
    pi_inv, sign_mask, eta, noise_limit, proj,
):
    params = state.params
    bs = state.batch_stats

    # empirical_ntk_vp_fn returns a callable f: R^{N_val×C} → R^{N_tr×C'}
    f_all = lambda X_in: state.apply_fn(
        {"params": params, "batch_stats": bs},
        X_in, train=False, mutable=False
    )
    ntk_vp_all = nt.empirical_ntk_vp_fn(f_all, X_tr, X_val, params)

    # compute factor = n_tr / sum_i [ π_i * sum_j sign_mask[i,j] * full_ones[j,i] ]
    C = f_all(X_tr[:1]).shape[-1]
    ones_val = jnp.ones((X_val.shape[0], C))
    full_ones = ntk_vp_all(ones_val)  # shape (n_tr, )
    sum_entries = jnp.sum(pi_inv * jnp.sum(sign_mask * full_ones, axis=1))
    factor = X_tr.shape[0] / sum_entries

    def signed_vp(u_val):
        # u_val: (n_val,)
        u_mat = jnp.tile(u_val[:, None], (1, C))
        full = ntk_vp_all(u_mat)  # shape (n_tr, C)
        weighted = pi_inv * jnp.sum(sign_mask * full, axis=1)
        return factor * weighted  # shape (n_tr,)
    
    @jax.jit
    def compute_delta_w(u_train, u_val):
       signed = signed_vp(u_val)  # returns (n_tr,)
       raw = u_train * signed
       centered = raw - jnp.mean(raw)
       clipped = jnp.minimum(centered, 0.0)
       return clipped

    u_train = compute_residuals(state.apply_fn, params, bs, X_tr, y_tr)
    u_val = compute_residuals(state.apply_fn, params, bs, X_val, y_val)
    delta_w = compute_delta_w(u_train, u_val)
    w_new = w - eta * delta_w

    # project onto top (n_tr - noise_rate * n_tr) coordinates
    if proj == "euclidean":
        w_proj, _ = project_euclidean(w_new, int(X_tr.shape[0] - noise_limit))
    else:  # 'lp'
        w_proj = project_lp(w_new, int(X_tr.shape[0] - noise_limit))

    return w_proj

def train_loop(
    rng,
    X_tr, y_tr, X_val, y_val, X_te, y_te,
    alpha0=0.5,
    eta=0.1,
    lr=0.1,
    batch_size=128,
    num_epochs=350,
    reweight_every=5,
    noise_rate=0.4,
    proj="lp",
):
    m = X_tr.shape[0]

    # 5.1) Initialize model, params, batch_stats, optimizer
    rng, init_key = jax.random.split(rng)
    model = ResNet18(num_classes=10)
    dummy = jnp.zeros((1, *X_tr.shape[1:]), dtype=jnp.float32)
    variables = model.init(init_key, dummy, train=True)
    params, batch_stats = variables["params"], variables["batch_stats"]

    lr_schedule = make_lr_schedule(m, batch_size, num_epochs, lr)
    decay_mask = make_decay_mask(params)  # <-- mirror params exactly
    tx = optax.chain(
        optax.add_decayed_weights(5e-4, mask=decay_mask),
        optax.sgd(learning_rate=lr_schedule, momentum=0.9),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
    )

    # 5.2) Precompute class reweight constants
    pi_inv = compute_pi_inv(y_tr)                  # shape (m,)
    C = jnp.max(y_tr) + 1                           # num_classes
    class_ids = jnp.arange(C)
    # sign_mask: shape (m, C), +1 where y_tr matches class, −alpha0 otherwise
    sign_mask = jnp.where(y_tr[:, None] == class_ids[None, :], 1.0, -alpha0)

    w = jnp.ones((m,))     # initial example weights
    w_history = [w]

    stats = {"tr_loss": [], "val_loss": [], "tr_acc": [], "te_acc": []}

    steps_per_epoch = math.ceil(m / batch_size)

    for epoch in tqdm(range(1, num_epochs + 1)):
        # — Shuffle & minibatches
        rng, sk = jax.random.split(rng)
        perm = jax.random.permutation(sk, m)
        Xs, ys, ws = X_tr[perm], y_tr[perm], w[perm]

        num_batches = int(jnp.ceil(m / batch_size))
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, m)
            xb, yb, wb = Xs[start:end], ys[start:end], ws[start:end]

            rng, subkey = jax.random.split(rng)
            xb_aug, rng = augment_batch(subkey, xb)

            state, _ = train_step(state, xb_aug, yb, wb)

        # — Reweight every few epochs (after warmup)
        if (epoch > 5) and (epoch % reweight_every == 0):
            w = reweight_step(
                state,
                X_tr, y_tr, w,
                X_val, y_val,
                pi_inv, sign_mask,
                eta, noise_rate * m, proj,
            )

        # — Logging every 10 epochs (or first)
        if (epoch == 1) or (epoch % 10 == 0):
            w_history.append(w)
            loss_tr, logits_tr = evaluate_in_batches(state, X_tr, y_tr, w, batch_size=512)
            loss_val, _ = evaluate_in_batches(state, X_val, y_val, jnp.ones_like(y_val), batch_size=512)
            logits_te = evaluate_in_batches(state, X_te, y_all=None, w_all=None, batch_size=512)

            tr_acc = jnp.mean(jnp.argmax(logits_tr, -1) == y_tr).item()
            te_acc = jnp.mean(jnp.argmax(logits_te, -1) == y_te).item()

            stats["tr_loss"].append(float(loss_tr))
            stats["val_loss"].append(float(loss_val))
            stats["tr_acc"].append(tr_acc)
            stats["te_acc"].append(te_acc)

            if (len(stats["val_loss"]) > 1) and (stats["val_loss"][-1] > stats["val_loss"][-2]):
                print(f"Epoch {epoch:02d}: val_loss increased; consider early stopping.")

            print(
                f"Epoch {epoch:02d} | "
                f"tr_loss={loss_tr:.4f}, val_loss={loss_val:.4f} | "
                f"tr_acc={tr_acc*100:.2f}%, te_acc={te_acc*100:.2f}% | "
                f"||Δw||={jnp.linalg.norm(w_history[-1] - w_history[-2]):.4f}"
            )

    return state.params, w, w_history, stats

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    X_tr, y_tr, y_tr_noisy, X_val, y_val, X_te, y_te = create_noisy_cifar(rng, subsample=(40000, 10000), noise_rate=0.4)
    
    params_final, w_final, w_list, stats0 = train_loop(
        rng, X_tr, y_tr_noisy, X_val, y_val, X_te, y_te,
        alpha0=0.5, eta=1.0, lr=0.1,
        batch_size=128, num_epochs=350, reweight_every=np.inf
    ) 