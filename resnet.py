import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import flax, optax

from functools import partial
from typing import Sequence, Callable, Any, Tuple

class _ResNetBlock(nn.Module):
    """ResNet V1 basic block."""
    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x

        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)

        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)

class _ResNet(nn.Module):
    """ResNet V1 generic."""
    stage_sizes: Sequence[int]
    block_cls: Any
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: Any = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        # initial conv / pool
        x = conv(self.num_filters, (7, 7), (2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = self.act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        # 4 stages
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if (i > 0 and j == 0) else (1, 1)
                x = self.block_cls(self.num_filters * 2**i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)

        # global average pool + classifier
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return jnp.asarray(x, self.dtype)

ResNet18 = partial(_ResNet, stage_sizes=[2, 2, 2, 2], block_cls=_ResNetBlock)

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict

def make_resnet18_stateful(num_classes: int = 10, lr: float = 0.1):
    model = ResNet18(num_classes=num_classes)

    def init_fn(rng, input_shape):
        # create a dummy batch of size 1
        dummy = jnp.zeros((1, *input_shape), jnp.float32)
        variables = model.init(rng, dummy, train=True)
        return TrainState.create(
            apply_fn       = model.apply,
            params         = variables['params'],
            batch_stats    = variables['batch_stats'],
            tx = optax.sgd(learning_rate=lr, momentum=0.9, weight_decay=5e-4),
        )

    def apply_fn(state, x, train: bool):
        # when train=True we want batch_stats to update, else we just read them
        mutable = ['batch_stats'] if train else False
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        out, updated = state.apply_fn(
            variables, x, train=train, mutable=mutable
        )
        if train:
            # updated is {'batch_stats': new_stats}
            state = state.replace(batch_stats=updated['batch_stats'])
        return out, state

    return init_fn, apply_fn