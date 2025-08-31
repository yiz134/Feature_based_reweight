import jax
import jax.numpy as jnp
from jax import lax

def _newton_step(state, closed):
    """One semi‑smooth Newton iteration."""
    it, tau, F = state
    v, S, lo, hi = closed

    w   = jnp.clip(v - tau, 0.0, 1.0)
    F   = w.sum() - S
    m   = jnp.sum((w > 0.0) & (w < 1.0))      # −F′(tau)

    inv_m = jnp.where(m == 0, 1.0, 1.0 / m)
    step  = jnp.where(m == 0,
                      jnp.sign(F),
                      F * inv_m)              #  = F / m

    tau = jnp.clip(tau + step, lo, hi)
    return (it + 1, tau, F)

def project_capped_simplex_ssn_jax(v, S, tol=1e-8, max_iter=50):
    """
    Semi‑smooth Newton projection onto { w∈[0,1]^n : sum(w)=S }.
    """
    v = jnp.asarray(v)
    n = v.shape[0]
    S = jnp.asarray(S, dtype=v.dtype)

    lo, hi = v.min() - 1.0, v.max()
    tau0   = (v.sum() - S) / n

    w0  = jnp.clip(v - tau0, 0.0, 1.0)
    F0  = w0.sum() - S
    state0  = (jnp.array(0, jnp.int32), tau0, F0)
    closed  = (v, S, lo, hi)

    cond = lambda s: jnp.logical_and(s[0] < max_iter, jnp.abs(s[2]) > tol)
    final_state = lax.while_loop(cond,
                                 lambda s: _newton_step(s, closed),
                                 state0)
    _, tau_star, _ = final_state
    w_star = jnp.clip(v - tau_star, 0.0, 1.0)

    # Degenerate sums handled branch‑less
    w_star = jnp.where(S <= 0.0,
                       jnp.zeros_like(v),
                       jnp.where(S >= n, jnp.ones_like(v), w_star))
    tau_star = jnp.where(S <= 0.0,
                         hi,
                         jnp.where(S >= n, lo, tau_star))
    return w_star, tau_star
project_euclidean = jax.jit(project_capped_simplex_ssn_jax)


@jax.jit
def project_lp(g: jnp.ndarray, S: float) -> jnp.ndarray:
    """
    Solve max <g,w> s.t. sum_i w_i = S, 0 <= w_i <= 1,
    by greedily filling budget using a scan over sorted indices.
    """
    # sort indices by descending score
    idx_sorted = jnp.argsort(-g)  # [n]

    def scan_step(carry, idx):
        remaining, w = carry
        # allocate at this index: either full 1 or the remaining budget
        alloc = jnp.minimum(remaining, 1.0)
        w = w.at[idx].set(alloc)
        remaining = remaining - alloc
        return (remaining, w), None

    # initialize remaining budget and zero weights
    init_carry = (S, jnp.zeros_like(g))
    (remaining_final, w_final), _ = lax.scan(scan_step, init_carry, idx_sorted)
    return w_final