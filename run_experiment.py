from typing import Iterator, Mapping, Tuple
from copy import deepcopy
import time
from absl import app
import haiku as hk
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from functools import partial
import math

Batch = Tuple[np.ndarray, np.ndarray]



CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def net_fn(batch: Batch) -> jnp.ndarray:
  """Convolution neural network.
  """
  x = normalize(batch[0])
  
  # Do NOT alter the architecture definition below.
  net = hk.Sequential([
      hk.Conv2D(output_channels=6*3, kernel_shape=(5,5)),
      jax.nn.relu,
      hk.AvgPool(window_shape=(2,2), strides=(2,2), padding='VALID'),
      jax.nn.relu,
      hk.Conv2D(output_channels=16*3, kernel_shape=(5,5)),
      jax.nn.relu,
      hk.AvgPool(window_shape=(2,2), strides=(2,2), padding='VALID'),
      hk.Flatten(),
      hk.Linear(3000), jax.nn.relu,
      hk.Linear(2000), jax.nn.relu,
      hk.Linear(2000), jax.nn.relu,
      hk.Linear(1000), jax.nn.relu,
      hk.Linear(10),
  ])
  return net(x)

def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
) -> Iterator[tuple]:
  """Loads the dataset as a generator of batches.

  Args:
    split: either train[split1:split2] or test

  Returns:
    An iterator with pairs (images, labels). 
    The images have shape (B, 32, 32, 3) 
    and the labels have shape (B, 10), where B is the batch_size.
  """
  ds = tfds.load('cifar10', split=split, as_supervised=True).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))

def compute_loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
  """Compute the loss of the network, including L2.

    Args:
      params: pytree representing the parameters
      batch: batch of data, (B, 32, 32, 3)

    Returns:
      calculated loss value
  """
  x, y = batch
  logits = net.apply(params, batch)
  labels = jax.nn.one_hot(y, 10)

  # TODO: add code below to compute the l2_loss variable
  l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
  weighted_l2_loss = 0.5 * l2_loss 
  
  softmax_xent = -jnp.mean(labels * jax.nn.log_softmax(logits)) #

  return softmax_xent + (1e-4 * weighted_l2_loss)  #


@jax.jit
def compute_accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
  """Compute model accuracy

  Args:
    params: pytree representing the model parameters
    batch: batch of data, shape: (B, 32, 32, 3)

  Returns:
    calculated accuracy value
  """  
  predictions = net.apply(params, batch)

  # TODO: add code below to compute the accuracy over the batch.
  accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch[1])  
  return accuracy

@jax.jit
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    batch: Batch,
) -> Tuple[hk.Params, optax.OptState]:
  """calculates new state and parameters using the the data

  Args:
    params: model's parameters
    opt_state: pytree representing an optimizer state
    batch: batch of data, shape: (B, 32, 32, 3)

  Returns:
    new_params: Updated parameters, with same structure, shape and type as `params`
    opt_state: updated state
  """
  grads = jax.grad(compute_loss)(params, batch)
  updates, opt_state = opt.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, opt_state

@jax.jit
def ema_update(params, avg_params):
  """Calculates the Exponential Moving Average for the latest parameters

  Args:
    params: latest values of model parameters.
    avg_params: moving average values of the model parameters.

  Returns:
    an updated moving average {step_size * params + (1-step_size)* avg_params} of the params.
  """
  return optax.incremental_update(params, avg_params, step_size=0.001)


def normalize(images):
  """Normalize the data

  Args:
    images: batch of image data, (B, 32, 32, 3).

  Returns:
    normalized batch of data, (B, 32, 32, 3).
  """
  mean = np.asarray(CIFAR10_MEAN)
  std = np.asarray(CIFAR10_STD)
  x = images.astype(jnp.float32) / 255.  # Bug
  x -= mean  # Bug
  x /= std
  return x

# Train model
net = hk.without_apply_rng(hk.transform(net_fn))

# Do not change learning rate
opt = optax.adam(1e-3)

train = load_dataset("train[0%:80%]", is_training=True, batch_size=64)   # Bug
validation = load_dataset("train[80%:]", is_training=False, batch_size=10000)  # Bug
test = load_dataset("test", is_training=False, batch_size=10000)

params = avg_params = net.init(jax.random.PRNGKey(42), next(train))
opt_state = opt.init(params)

# Do not alter the number of steps
for step in range(10001):
  if step % 1000 == 0:
    val_accuracy = compute_accuracy(avg_params, next(validation))
    test_accuracy = compute_accuracy(avg_params, next(test))
    val_accuracy, test_accuracy = jax.device_get(
        (val_accuracy, test_accuracy))
    print(f"[Step {step}] Validation / Test accuracy: "
          f"{val_accuracy:.3f} / {test_accuracy:.3f}.")

  params, opt_state = update(params, opt_state, next(train))
  avg_params = ema_update(params, avg_params)