# Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi

import torch
import torch.nn as nn
import math
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
  def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
    super(GCN, self).__init__()
    self.g = g
    self.layers = nn.ModuleList()
    self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
    for i in range(n_layers - 1):
      self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
    self.layers.append(GraphConv(n_hidden, n_classes))
    self.dropout = nn.Dropout(p=dropout)
    
  def forward(self, features):
    h = features
    for i, layer in enumerate(self.layers):
      if i != 0:
        h = self.dropout(h)
      h = layer(self.g, h)
    return h
  
# E = Encoder, gcn = GCN(g)
# E(X) = gcn(X) for positive example
# E(X) = gcn(randomperm(X)) for negative example
class Encoder(nn.Module):
  def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
    super(Encoder, self).__init__()
    self.g = g
    self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
    
  def forward(self, features, corrupt=False):
    if corrupt:
      perm = torch.randperm(self.g.number_of_nodes())
      features = features[perm]
    features = self.conv(features)
    return features
  
# D(H, s) = H * W * s  
class Distriminator(nn.Module):
  def __init__(self, n_hidden):
    super(Discriminator, self).__init__()
    self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
    self.reset_parameters()
    
  def reset_parameters(self):
    size = self.weight.size(0)
    stdv = 1.0 / math.sqrt(size)
    if self.weight is not None:
      self.weight.uniform_(-stdv, stdv)
      
  def forward(self, features, summary):
    features = torch.matmul(features, torch.matmul(self.weight, summary))
    return features
  

class DGI(nn.Module):
  def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
    super(DGI, self).__init__()
    self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
    self.discriminator = Distriminator(n_hidden)
    self.loss = nn.BCEWithLogitsLoss()
    
  def forward(self, features):
    positive = self.encoder(features, corrupt=False)
    negative = self.encoder(features, corrupt=True)
    summary = torch.sigmoid(positive.mean(dim=0))
    positive = self.discriminator(positive, summary)
    negative = self.discriminator(negative, summary)
    
    l1 = self.loss(positive, torch.ones_like(positive))
    l2 = self.loss(negative, torch.zeros_like(negative))
    
    return l1 + l2
