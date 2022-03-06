# Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn
improt math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gcn_msg(edges):
  msg = edges.src['h'] * edges.src['norm']
  return {'m': msg}

def gcn_reduce(nodes):
  h = torch.sum(nodes.mailbox['m'], 1) * nodes.data['norm']
  return {'h': h}

class GCNLayer(nn.Module):
  def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
    super(GCNLayer, self).__init__()
    self.g = g
    self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
    self.activation = activation
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_feats))
    if dropout:
      self.dropout = nn.Dropout(p=dropout)
    self.reset_parameters()
    
  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if bias:
      self.bias.data.uniform_(-stdv, stdv)
    
  def forward(self, h):
    if self.dropout:
      h = self.dropout(h)
    self.g.ndata['h'] = torch.mm(h, self.weight)
    self.g.update_all(gcn_msg, gcn_reduce)
    h = self.g.ndata['h']
		if self.bias:
			h = h + self.bias
		if self.activation:
			h = self.activation(h)
    h = self.g.ndata.pop('h')
		return h
