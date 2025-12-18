from typing import Tuple

import jax
import jax.numpy as jnp
from einops import reduce

i32 = jnp.int32
f32 = jnp.float32
sg = jax.lax.stop_gradient

def symlog(x:jax.Array):
    return jax.lax.stop_gradient(jnp.sign(x)*jnp.log1p(jnp.abs(x)))

def symexp(x:jax.Array):
    return jax.lax.stop_gradient(jnp.sign(x)*jnp.expm1(jnp.abs(x)))

class SymLogTwoHot:
    def __init__(self, bins:int, min_val:float, max_val:float):
        self.bins = jnp.linspace(min_val, max_val, bins, dtype=jnp.float32)
        self.min_val = min_val
        self.max_val = max_val
        self.bin_length = (max_val-min_val)/(bins-1)

    def encode(self, scalar:jax.Array):
        scalar = symlog(scalar)
        below = (self.bins <= scalar[...,None]).sum(-1) - 1
        above = self.bins.shape[-1] - (self.bins > scalar[...,None]).sum(-1)
        below = jnp.clip(below, min=0, max=self.bins.shape[-1]-1)
        above = jnp.clip(above, min=0, max=self.bins.shape[-1]-1)
        equal = (below==above)
        diff_below = jnp.where(equal, 1, jnp.abs(scalar - self.bins[below]))
        diff_above = jnp.where(equal, 1, jnp.abs(scalar - self.bins[above]))
        total = diff_above + diff_below
        below_weight = diff_above/total
        above_weight = diff_below/total
        twohot = below_weight[...,None]*jax.nn.one_hot(below, self.bins.shape[-1], dtype=jnp.float32) + \
                 above_weight[...,None]*jax.nn.one_hot(above, self.bins.shape[-1], dtype=jnp.float32)
        return twohot
    
    def decode(self, scalar:jax.Array):
        weight_sum = jnp.einsum('...k,k -> ...',scalar,self.bins)
        return symexp(weight_sum)
    
    def compute_loss(self, target:jax.Array, logits:jax.Array):
      target_weights = self.encode(sg(target))
      loss = -target_weights*jax.nn.log_softmax(logits, axis=-1)
      loss = jnp.mean(jnp.sum(loss, axis=-1))
      return loss
      

class Output:

  def __repr__(self):
    name = type(self).__name__
    pred = self.pred()
    return f'{name}({pred.dtype}, shape={pred.shape})'

  def pred(self):
    raise NotImplementedError

  def loss(self, target):
    return -self.logp(sg(target))

  def sample(self, seed, shape=()):
    raise NotImplementedError

  def logp(self, event):
    raise NotImplementedError

  def prob(self, event):
    return jnp.exp(self.logp(event))

  def entropy(self):
    raise NotImplementedError

  def kl(self, other):
    raise NotImplementedError

class Agg(Output):

  def __init__(self, output, dims, agg=jnp.sum):
    self.output = output
    self.axes = [-i for i in range(1, dims + 1)]
    self.agg = agg

  def __repr__(self):
    name = type(self.output).__name__
    pred = self.pred()
    dims = len(self.axes)
    return f'{name}({pred.dtype}, shape={pred.shape}, agg={dims})'

  def pred(self):
    return self.output.pred()

  def loss(self, target):
    loss = self.output.loss(target)
    return self.agg(loss, self.axes)

  def sample(self, seed, shape=()):
    return self.output.sample(seed, shape)

  def logp(self, event):
    return self.output.logp(event).sum(self.axes)

  def prob(self, event):
    return self.output.prob(event).sum(self.axes)

  def entropy(self):
    entropy = self.output.entropy()
    return self.agg(entropy, self.axes)

  def kl(self, other):
    assert isinstance(other, Agg), other
    kl = self.output.kl(other.output)
    return self.agg(kl, self.axes)

class Normal(Output):

    def __init__(self, mean, stddev=1.0):
        self.mean = f32(mean)
        self.stddev = jnp.broadcast_to(f32(stddev), self.mean.shape)

    def pred(self):
        return self.mean

    def sample(self, seed, shape=()):
        sample = jax.random.normal(seed, shape + self.mean.shape, f32)
        return sample * self.stddev + self.mean

    def logp(self, event):
        assert jnp.issubdtype(event.dtype, jnp.floating), event.dtype
        return jax.scipy.stats.norm.logpdf(f32(event), self.mean, self.stddev)

    def entropy(self):
        return 0.5 * jnp.log(2 * jnp.pi * jnp.square(self.stddev)) + 0.5

    def kl(self, other):
        assert isinstance(other, type(self)), (self, other)
        return 0.5 * (
            jnp.square(self.stddev / other.stddev) +
            jnp.square(other.mean - self.mean) / jnp.square(other.stddev) +
            2 * jnp.log(other.stddev) - 2 * jnp.log(self.stddev) - 1)
    

def bounded_normal(minstd, maxstd, mean_std:Tuple[jax.Array]) -> Normal:
    mean,stddev = mean_std
    lo, hi = minstd, maxstd
    stddev = (hi - lo) * jax.nn.sigmoid(stddev + 2.0) + lo
    output = Normal(jnp.tanh(mean), stddev)
    output.minent = Normal(jnp.zeros_like(mean), minstd).entropy()
    output.maxent = Normal(jnp.zeros_like(mean), maxstd).entropy()
    return output

class Percentile:

    rate: float = 0.01
    limit: float = 1.
    perclo: float = 5.
    perchi: float= 95.
    debias: bool = False

    def __init__(self):
        self.lo = jnp.zeros((),dtype=jnp.float32)
        self.hi = jnp.zeros((),dtype=jnp.float32)
    
    def __call__(self,x,update):
        if update:
            self.update(x)
        return self.stats()
    
    def stats(self):
        return jax.lax.stop_gradient(jnp.maximum(self.limit, self.hi - self.lo))
    
    def update(self,x):
        x = jax.lax.stop_gradient(jnp.float32(x))
        self._update(x)
    
    def _update(self,x):
        lo = jnp.percentile(x,5.)
        hi = jnp.percentile(x,95.)
        self.lo = self.rate*lo + (1-self.rate)*self.lo
        self.hi = self.rate*hi + (1-self.rate)*self.hi


class Categorical(Output):

  def __init__(self, logits, unimix=0.0):
    logits = f32(logits)
    if unimix:
      probs = jax.nn.softmax(logits, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - unimix) * probs + unimix * uniform
      logits = jnp.log(probs)
    self.logits = logits

  def pred(self):
    return jnp.argmax(self.logits, -1)

  def sample(self, seed, shape=()):
    return jax.random.categorical(
        seed, self.logits, -1).squeeze(0)

  def logp(self, event):
    onehot = jax.nn.one_hot(event, self.logits.shape[-1])
    return (jax.nn.log_softmax(self.logits, -1) * onehot).sum(-1)

  def entropy(self):
    logprob = jax.nn.log_softmax(self.logits, -1)
    prob = jax.nn.softmax(self.logits, -1)
    entropy = -(prob * logprob).sum(-1)
    return entropy

  def kl(self, other):
    logprob = jax.nn.log_softmax(self.logits, -1)
    logother = jax.nn.log_softmax(other.logits, -1)
    prob = jax.nn.softmax(self.logits, -1)
    return (prob * (logprob - logother)).sum(-1)

class Binary(Output):

  def __init__(self, logit):
    self.logit = f32(logit)

  def pred(self):
    return (self.logit > 0)

  def logp(self, event):
    event = f32(event)
    logp = jax.nn.log_sigmoid(self.logit).squeeze(axis=-1)
    lognotp = jax.nn.log_sigmoid(-self.logit).squeeze(axis=-1)
    return event * logp + (1 - event) * lognotp

  def sample(self, seed, shape=()):
    prob = jax.nn.sigmoid(self.logit)
    return jax.random.bernoulli(seed, prob, -1, shape + self.logit.shape)

def ImageMSE(pred:jax.Array, target:jax.Array):
  target = jax.device_put(target)
  target = sg(target.astype(jnp.float32)/255.)
  loss = (pred-target)**2
  loss = reduce(loss, 'B L H W C -> B L','sum')
  # loss = jnp.sum(loss,axis=[-1,-2,-3])
  return loss.mean()    

        
if __name__ == '__main__':
    # Twohot = SymLogTwoHot(255,-20,20)
    # scalar = jnp.arange(0,27).reshape(3,3,3)
    # twohot = Twohot.encode(scalar)
    # print(twohot)
    # print(Twohot.bins)
    # print(Twohot.decode(twohot))
    # randn = jax.random.randint(jax.random.PRNGKey(0),(2,3,4),0,3)
    # print(randn)
    # print(jnp.sum(randn,axis=[-1,-2]))
    
    pass