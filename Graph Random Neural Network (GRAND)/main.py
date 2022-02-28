import numpy as np 
import torch as th 
import torch.nn as nn 
import dgl.function as fn 
import torch.nn.functional as F

def drop_node(feats, drop_rate, training):
  if training:
		n = feats.shape[0]
    keep_rate = 1. - drop_rate
		keep_rates = th.FloatTensor(np.full(n, keep_rate))
		masks = th.bernoulli(keep_rates).unsqueeze(1)	        # shape: (n, 1)
		feats = masks.to(feats.device) * feats / keep_rate
	return feats
