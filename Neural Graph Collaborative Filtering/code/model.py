# Source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/NGCF/NGCF/model.py

import torch
from torch import nn
import torch.nn.functional as F
import dgl.function as fn


class NGCFLayer(nn.Module):
  def __init__(self, in_size, out_size, norm_dict, activation=nn.LeakyReLu(0.2), dropout=0.):
    """
    norm_dict is a dict whose keys are g.canonical_etypes.
    norm_dict[(srctype, etype, dsttype)] is a tensor of shape (N, 1), where N = # of the canonical edge type.
    """
    super(NGCFLayer, self).__init__()
    self.in_size = in_size
    self.out_size = out_size
    self.W1 = nn.Linear(in_size, out_size, bias=True)
    self.W2 = nn.Linear(in_size, out_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
    self.norm_dict = norm_dict
    
    # initialization
    nn.init.xavier_uniform_(self.W1.weight)
    nn.init.xavier_uniform_(self.W2.weight)
    nn.init.constant_(self.W1.bias, 0.)
    nn.init.constant_(self.W2.bias, 0.)
    
  def forward(self, g, feat_dict):
    """
    feat_dict is a dict whose keys are node types.
    feat_dict[node_type_str] is a tensor of shape (N, d) where N = # of nodes of the node type and d = embedding_dimension
    ex) feat_dict['user'] is [e_{u_1}, ..., e_{u_N}]^T, and feat_dict['item'] is [e_{i_1}, ..., e_{i_M}]^T
    Note E = [e_{u_1}, ..., e_{u_N}, e_{i_1}, ..., e_{i_M}]^T in the paper.
    """
    funcs = {}
    for canonical_etype in g.canonical_etypes:
      srctype, etype, dsttype = canonical_etype
      if srctype == dsttype: # for self loops.
        messages = self.W1(feat_dict[srctype])
        g.nodes[srctype].data[etype] = messages   # store in ndata
        funcs[canonical_etype] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))
      else:
        src, dst = g.edges(etype=canonical_etype)
        norm = self.norm_dict[canonical_etype]
        messages = self.W1(feat_dict[srctype][src]) + self.W2(feat_dict[srctype][src] * feat_dict[dsttype][dst])
        messages = norm * messages
        g.edges[canonical_etype].data[etype] = messages   # store in edata
        funcs[canonical_etype] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))
        
    g.multi_update_all(funcs, 'sum')
    
    feature_dict = {}
    for ntype in g.ntypes:
      h = self.activation(g.nodes[ntype].data['h'])
      h = self.dropout(h)
      h = F.normalize(h, dim=1, p=2)  # l2 normalize
      feature_dict[ntype] = h
      
    return feature_dict
        
        
class NGCF(nn.Module):
  
