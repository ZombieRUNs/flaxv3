import math
import re

import jax
import jax.numpy as jnp
import ninjax as nj
import optax

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient



def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)

def make_opt(
    lr: float = 4e-5,
    agc: float = 0.3,
    eps: float = 1e-20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    momentum: bool = True,
    nesterov: bool = False,
    wd: float = 0.0,
    wdregex: str = r'/kernel$',
    schedule: str = 'const',
    warmup: int = 1000,
    anneal: int = 0,
):
  chain = []
  chain.append(clip_by_agc(agc))
  chain.append(scale_by_rms(beta2, eps))
  chain.append(scale_by_momentum(beta1, nesterov))
  if wd:
    assert not wdregex[0].isnumeric(), wdregex
    pattern = re.compile(wdregex)
    wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
    chain.append(optax.add_decayed_weights(wd, wdmask))
  assert anneal > 0 or schedule == 'const'
  if schedule == 'const':
    sched = optax.constant_schedule(lr)
  elif schedule == 'linear':
    sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
  elif schedule == 'cosine':
    sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
  else:
    raise NotImplementedError(schedule)
  if warmup:
    ramp = optax.linear_schedule(0.0, lr, warmup)
    sched = optax.join_schedules([ramp, sched], [warmup])
  chain.append(optax.scale_by_learning_rate(sched))
  return optax.chain(*chain)

def make_simple_opt(lr:float=4e-5, grad_norm:float=100.):
  return optax.chain(
   optax.clip_by_global_norm(grad_norm),
   optax.adam(lr,eps=1e-5),)
