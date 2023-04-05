from typing import Callable, Iterator, Mapping, Sequence, Tuple
from absl import app
import functools

Batch = Mapping[str, np.ndarray]
Predicate = Callable[[str, str, jnp.ndarray], bool]
PredicateMap = Mapping[Predicate, jnp.ndarray]
ModuleSparsity = Sequence[Tuple[Predicate, jnp.ndarray]]


def topk_mask(value: jnp.ndarray, density_fraction: float) -> jnp.ndarray:
  def topk_mask_internal(value):
    assert value.ndim == 1
    indices = jnp.argsort(value)
    k = jnp.round(density_fraction * jnp.size(value)).astype(jnp.int32)
    mask = jnp.greater_equal(np.arange(value.size), value.size - k)
    mask = jnp.zeros_like(mask).at[indices].set(mask)
    return mask.astype(np.int32)

  # shuffle value so that identical values aren't always pruned
  # with a bias to lower indices
  orig_shape = value.shape
  value = jnp.reshape(value, -1)
  shuffled_indices = jax.random.shuffle(
      jax.random.PRNGKey(42), jnp.arange(0, jnp.size(value), dtype=jnp.int32))

  shuffled_mask = topk_mask_internal(value[shuffled_indices])
  mask = jnp.zeros_like(shuffled_mask).at[shuffled_indices].set(shuffled_mask)
  mask = jnp.reshape(mask, orig_shape)
  return mask


def zhugupta_func(progress: jnp.ndarray) -> jnp.ndarray:
  """From 'To Prune or Not To Prune' :cite:`zhu2017prune`."""
  return 1. - (1. - progress)**3


def _create_partitions(
    module_sparsity: ModuleSparsity, params: hk.Params
) -> Tuple[Sequence[hk.Params], Sequence[jnp.ndarray], hk.Params]:

  list_of_trees = []
  sparsity_list = []

  tail = params
  # Greedily match so that no parameter can be matched more than once
  for predicate, sparsity in module_sparsity:
    head, tail = hk.data_structures.partition(predicate, tail)
    list_of_trees.append(head)
    sparsity_list.append(sparsity)

  return list_of_trees, sparsity_list, tail


def sparsity_ignore(m: str, n: str, v: jnp.ndarray) -> bool:
  """Any parameter matching these conditions should generally not be pruned."""
  # n == 'b' when param is a bias
  return n == "b" or v.ndim == 1 or "batchnorm" in m or "batch_norm" in m


@functools.partial(jax.jit, static_argnums=2)
def apply_mask(params: hk.Params, masks: Sequence[hk.Params],
               module_sparsity: ModuleSparsity) -> hk.Params:

  params_to_prune, _, params_no_prune = _create_partitions(
      module_sparsity, params)
  pruned_params = []
  for value, mask in zip(params_to_prune, masks):
    pruned_params.append(
        jax.tree_util.tree_map(lambda x, y: x * y, value, mask))
  params = hk.data_structures.merge(*pruned_params, params_no_prune)
  return params


@functools.partial(jax.jit, static_argnums=2)
def update_mask(params: hk.Params, sparsity_fraction: float,
                module_sparsity: ModuleSparsity) -> Sequence[hk.Params]:
  """Generate masks based on module_sparsity and sparsity_fraction."""
  params_to_prune, sparsities, _ = _create_partitions(module_sparsity, params)
  masks = []

  def map_fn(x: jnp.ndarray, sparsity: float) -> jnp.ndarray:
    return topk_mask(jnp.abs(x), 1. - sparsity * sparsity_fraction)

  for tree, sparsity in zip(params_to_prune, sparsities):
    map_fn_sparsity = functools.partial(map_fn, sparsity=sparsity)
    mask = jax.tree_util.tree_map(map_fn_sparsity, tree)
    masks.append(mask)
  return masks


@jax.jit
def get_sparsity(params: hk.Params):
  """Calculate the total sparsity and tensor-wise sparsity of params."""
  total_params = sum(jnp.size(x) for x in jax.tree_util.tree_leaves(params))
  total_nnz = sum(jnp.sum(x != 0.) for x in jax.tree_util.tree_leaves(params))
  leaf_sparsity = jax.tree_util.tree_map(
      lambda x: jnp.sum(x == 0) / jnp.size(x), params)
  return total_params, total_nnz, leaf_sparsity

@jax.jit
def get_updates(
    params: hk.Params,
    opt_state: optax.OptState,
    batch: Batch,
) -> Tuple[hk.Params, optax.OptState]:
  """Learning rule (stochastic gradient descent)."""
  grads = jax.grad(compute_loss)(params, batch)
  updates, opt_state = opt.update(grads, opt_state)
  return updates, opt_state


def smaller_net_fn(batch: Batch) -> jnp.ndarray:
  CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
  CIFAR10_STD = (0.2023, 0.1994, 0.2010)
  mean = np.asarray(CIFAR10_MEAN)
  std = np.asarray(CIFAR10_STD)

  x = batch[0].astype(jnp.uint8)/255  # 1. float32
  x -= mean # 2.
  x /= std

  net = hk.Sequential([
      hk.Conv2D(output_channels=6*3, kernel_shape=(5,5)),
      jax.nn.relu,
      hk.AvgPool(window_shape=(2,2), strides=(2,2), padding='VALID'),
      jax.nn.relu,
      hk.Conv2D(output_channels=16*3, kernel_shape=(5,5)),
      jax.nn.relu,
      hk.AvgPool(window_shape=(2,2), strides=(2,2), padding='VALID'),
      #jax.nn.relu,
      #hk.Conv2D(output_channels=16*3, kernel_shape=(5,5)),
      #hk.AvgPool(window_shape=(2,2), strides=(2,2), padding='VALID'),
      hk.Flatten(),
      hk.Linear(512), jax.nn.relu,
      hk.Linear(10),
  ])
  return net(x)