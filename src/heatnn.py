import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import haiku as hk
from haiku_geometric.nn import GATConv, GCNConv
from haiku_geometric.nn.pool import global_mean_pool
from haiku_geometric.datasets.base import DataGraphTuple
from haiku_geometric.utils import batch as gbatch
from haiku_geometric.utils import unbatch

class Representation(hk.Module):
  def __init__(self, n_node, n_features, embedding_dim, max_neighbors, name='representation'):
    super().__init__(name=name)

    self.n_node = n_node
    self.n_features = n_features
    self.embedding_dim = embedding_dim

    self.n_edge = max_neighbors * (n_node - 1)
    self.sleft = self.n_node * self.n_features

    self.gc1 = GATConv(4*self.embedding_dim, add_self_loops = False)
    self.ln1 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.gc2 = GATConv(2*self.embedding_dim, add_self_loops = False)
    self.ln2 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.gc3 = GATConv(2*self.embedding_dim, add_self_loops = False)
    self.ln3 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.gc4 = GATConv(self.embedding_dim, add_self_loops = False)

  def __call__(self, r):

    # podaci flattenani u array zbog paralelizma pretraživanja stabla
    # TODO
    # ovo bi se moglo ubrzati
    # u mctx/_src/tree.py dodati uz `embeddings`: `senders` i `receivers`
    # i sukladno tome prosljeđivati
    obs = [DataGraphTuple(
            nodes=x[:self.sleft].reshape(self.n_node, self.n_features),
            edges=None,
            receivers=x[self.sleft+self.n_edge:self.sleft+2*self.n_edge].astype(int),
            senders=x[self.sleft:self.sleft+self.n_edge].astype(int),
            n_node=jnp.array([self.n_node]),
            n_edge=jnp.array([self.n_edge]),
            globals=None,
            position=None,
            y=None,
            train_mask=None
        ) for x in r]
    obs = gbatch(obs)
    # jax.debug.print("obs+g {x}", x=obs)
    # jax.debug.print("obs {x}", x=obs[0])
    s = self.gc1(nodes=obs[0],
                 senders=obs[3],
                 receivers=obs[2])
    s = self.ln1(s.reshape(-1, self.n_node, self.gc1.out_channels)).reshape(-1, self.gc1.out_channels)
    s = jax.nn.relu(s)
    s = self.gc2(nodes=s,
                 senders=obs[3],
                 receivers=obs[2])
    s = self.ln2(s.reshape(-1, self.n_node, self.gc2.out_channels)).reshape(-1, self.gc2.out_channels)
    s = jax.nn.relu(s)
    s = self.gc3(nodes=s,
                 senders=obs[3],
                 receivers=obs[2])
    s = self.ln3(s.reshape(-1, self.n_node, self.gc3.out_channels)).reshape(-1, self.gc3.out_channels)
    s = jax.nn.relu(s)
    s = self.gc4(nodes=s,
                 senders=obs[3],
                 receivers=obs[2])
    
    # jax.debug.print("embedding {x}", x=s)

    r = jnp.concatenate((s.reshape(-1, self.n_node*self.embedding_dim), r[:, self.n_node*self.n_features:]), axis=1)
    # jax.debug.print("emb+g {x}", x=r)

    # jax.debug.print("emb.sum: {r}", r=jnp.sum(r))

    return r
  
class Prediction(hk.Module):
  def __init__(self, full_support_size, n_node, embedding_dim, max_neighbors, name='prediction'):
    super().__init__(name=name)        
    self.full_support_size = full_support_size
    self.n_node = n_node
    self.embedding_dim = embedding_dim

    self.n_edge = max_neighbors * (n_node - 1)
    self.sleft = self.n_node * self.embedding_dim

    # 1 brid 1 odluka
    self.pi_gc1 = GCNConv(1, add_self_loops = False)
    self.pi_ln1 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.pi_gc2 = GCNConv(1, add_self_loops = False)

    self.v_gc1 = GCNConv(full_support_size, add_self_loops = False)
    self.v_ln1 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.v_gc2 = GCNConv(full_support_size, add_self_loops = False)
  
  def __call__(self, s):
    s = [DataGraphTuple(
            nodes=x[:self.sleft].reshape(self.n_node, self.embedding_dim),
            edges=None,
            receivers=x[self.sleft+self.n_edge:self.sleft+2*self.n_edge].astype(int),
            senders=x[self.sleft:self.sleft+self.n_edge].astype(int),
            n_node=jnp.array([self.n_node]),
            n_edge=jnp.array([self.n_edge]),
            globals=None,
            position=None,
            y=None,
            train_mask=None
        ) for x in s]
    s = gbatch(s)
    # jax.debug.print("pred ns {x}", x=s[0])
    # jax.debug.print("pred ns+g {x}", x=s)


    logits = self.pi_gc1(s[0], s[3], s[2])
    # jax.debug.print("logits.sum: {logits}", logits=logits.shape)
    logits = self.pi_ln1(logits.reshape(-1, self.n_node, 1)).reshape(-1, 1)
    # jax.debug.print("logits.sum: {logits}", logits=jnp.sum(logits))
    logits = jax.nn.relu(logits)
    # jax.debug.print("logits.sum: {logits}", logits=jnp.sum(logits))
    logits = self.pi_gc2(logits, s[3], s[2]).reshape(-1, self.n_node)[:, :-2]
    # jax.debug.print("logits.sum: {logits}", logits=jnp.sum(logits))

    v = self.v_gc1(s[0], s[3], s[2])
    v = self.v_ln1(v.reshape(-1, self.n_node, self.full_support_size)).reshape(-1, self.full_support_size)
    v = jax.nn.relu(v)
    v = self.v_gc2(v, s[3], s[2])
    # readout
    v = jnp.sum(v.reshape(-1, self.n_node, self.full_support_size), axis=(1,))
    # jax.debug.print("v.sum: {v}", v=jnp.sum(v))
    # jax.debug.print("logits.sum: {logits}", logits=jnp.sum(logits))
    # jax.debug.print("value {v} \n logits {l}", v = v, l = logits)
    return v, logits
  
class Dynamic(hk.Module):
  def __init__(self, num_actions, full_support_size, n_node, embedding_dim, max_neighbors, name='dynamic'):
    super().__init__(name=name)        
    self.full_support_size = full_support_size
    self.n_node = n_node
    self.embedding_dim = embedding_dim

    self.n_edge = max_neighbors * (n_node - 1)
    self.sleft = self.n_node * self.embedding_dim # početak sendersa u flattenanom arrayju

    self.num_actions = num_actions

    self.gc1 = GATConv(4*embedding_dim, add_self_loops = False)
    self.ln1 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.gc2 = GATConv(2*embedding_dim, add_self_loops = False)
    self.ln2 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.gc3 = GATConv(2*embedding_dim, add_self_loops = False)
    self.ln3 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.gc4 = GATConv(embedding_dim, add_self_loops = False)

    self.r_gc1 = GATConv(full_support_size, add_self_loops = False)
    self.r_ln1 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.r_gc2 = GATConv(full_support_size, add_self_loops = False)
    self.r_ln2 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.r_gc3 = GATConv(full_support_size, add_self_loops = False)
    self.r_ln3 = hk.LayerNorm(axis=(-2), create_scale=True, create_offset=True)
    self.r_gc4 = GATConv(full_support_size, add_self_loops = False)
  
  def __call__(self, ns, a):
    s = [DataGraphTuple(
            nodes=x[:self.sleft].reshape(self.n_node, self.embedding_dim),
            edges=None,
            receivers=x[self.sleft+self.n_edge:self.sleft+2*self.n_edge].astype(int),
            senders=x[self.sleft:self.sleft+self.n_edge].astype(int),
            n_node=jnp.array([self.n_node]),
            n_edge=jnp.array([self.n_edge]),
            globals=None,
            position=None,
            y=None,
            train_mask=None
        ) for x in ns]
    s = gbatch(s)
    # jax.debug.print("dyn ns {x}", x=s[0])
    # jax.debug.print("dyn ns+g {x}", x=s)
    sa = jnp.hstack([s[0], jax.nn.one_hot(a, self.num_actions+1).reshape(-1, 1)])
    # jax.debug.print("dyn sa {x}", x=sa)

    r = self.r_gc1(sa, s[3], s[2])
    r = self.r_ln1(r.reshape(-1, self.n_node, self.full_support_size)).reshape(-1, self.full_support_size)
    r = jax.nn.relu(r)
    r = self.r_gc2(r, s[3], s[2])
    r = self.r_ln2(r.reshape(-1, self.n_node, self.full_support_size)).reshape(-1, self.full_support_size)
    r = jax.nn.relu(r)
    r = self.r_gc3(r, s[3], s[2])
    r = self.r_ln3(r.reshape(-1, self.n_node, self.full_support_size)).reshape(-1, self.full_support_size)
    r = jax.nn.relu(r)
    r = self.r_gc4(r, s[3], s[2])

    # readout
    r = jnp.sum(r.reshape(-1, self.n_node, self.full_support_size), axis=(1,))

    features = self.gc1(nodes=sa,
                 senders=s[3],
                 receivers=s[2])
    features = self.ln1(features.reshape(-1, self.n_node, 
                                         self.gc1.out_channels)).reshape(-1, self.gc1.out_channels)    
    features = jax.nn.relu(features)
    features = self.gc2(nodes=features,
                 senders=s[3],
                 receivers=s[2])
    features = self.ln2(features.reshape(-1, self.n_node, 
                                         self.gc2.out_channels)).reshape(-1, self.gc2.out_channels) 
    features = jax.nn.relu(features)
    features = self.gc3(nodes=features,
                 senders=s[3],
                 receivers=s[2])
    features = self.ln3(features.reshape(-1, self.n_node, 
                                         self.gc3.out_channels)).reshape(-1, self.gc3.out_channels) 
    features = jax.nn.relu(features)
    features = self.gc4(nodes=features,
                 senders=s[3],
                 receivers=s[2])
    
    features = features.reshape(-1, self.n_node*self.embedding_dim)
    # jax.debug.print("dyn r {x}", x=r)
    # jax.debug.print("dyn ns {x}", x=features)

    ns = jnp.concatenate((features, ns[:, self.n_node*self.embedding_dim:]), axis=1)
    # jax.debug.print("dyn emb {d} reward {r}", d=features, r=r)
    # jax.debug.print("r.sum: {r}", r=jnp.sum(r))
    # jax.debug.print("ns.sum: {ns}", ns=jnp.sum(ns))
    # jax.debug.print("dyn ns+g {x}", x=ns)
    return r, ns

def _init_representation_func(representation_module, n_node, n_features, embedding_dim, max_neighbors):
  def representation_func(obs):
    repr_model = representation_module(n_node, n_features, embedding_dim, max_neighbors)
    return repr_model(obs)
  return representation_func
  
def _init_prediction_func(prediction_module, full_support_size, n_node, embedding_dim, max_neighbors):
  def prediction_func(s):
    pred_model = prediction_module(full_support_size, n_node, embedding_dim, max_neighbors)
    return pred_model(s)
  return prediction_func

def _init_dynamic_func(dynamic_module, num_actions, full_support_size, n_node, embedding_dim, max_neighbors):
  def dynamic_func(s, a):
    dy_model = dynamic_module(num_actions, full_support_size, n_node, embedding_dim, max_neighbors)
    return dy_model(s, a)
  return dynamic_func 

