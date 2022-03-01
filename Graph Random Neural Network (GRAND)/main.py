import numpy as np 
import torch as th 
import torch.nn as nn 
import dgl.function as fn 
import torch.nn.functional as F

import argparse
import torch.optim as optim
import torch.nn as nn

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


def drop_node(feats, drop_rate):
	n = feats.shape[0]
	keep_rate = 1. - drop_rate
	keep_rates = th.FloatTensor(np.full(n, keep_rate))
	masks = th.bernoulli(keep_rates).unsqueeze(1)	        # shape: (n, 1)
	feats = masks.to(feats.device) * feats / keep_rate
	return feats


class MLP(nn.Module):
	def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
		super(MLP, self).__init__()
		self.layer1 = nn.Linear(nfeat, nhid)
		self.layer2 = nn.Linear(nhid, nclass)
		self.input_dropout = nn.Dropout(input_droprate)
		self.hidden_dropout = nn.Dropout(hidden_droprate)
		self.bn1 = nn.BatchNorm1d(nfeat)
		self.bn2 = nn.BatchNorm1d(nhid)
		self.use_bn = use_bn
		self.reset_parameters()
		
	def reset_parameters(self):
		self.layer1.reset_parameters()
		self.layer2.reset_parameters()
		
	def forward(self, x):
		if self.use_bn:
			x = self.bn1(x)
		x = self.input_dropout(x)
		x = F.relu(self.layer1(x))
		
		if self.use_bn:
			x = self.bn2(x)
		x = self.hidden_dropout(x)
		x = self.layer2(x)
		
		return x	# Note: output is raw
	
	
def GRANDConv(graph, feats, order):	# propagation
	with graph.local_scope():
		degs = graph.in_degrees().float().clamp(min=1)
		norm = th.pow(degs, -0.5).to(feats.device).unsqueeze(1)		# shape: (n, 1)
		graph.ndata['norm'] = norm
		graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

		x = feats
		y = feats + 0	# deep copy
		for i in range(order):
			graph.ndata['h'] = x
			graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
			x = graph.ndata.pop('h')
			y.add_(x)
			
	return y / (order + 1)


class GRAND(nn.Module):
	def __init__(self, in_dim, hid_dim, n_class, S=1, K=3, node_dropout=0., input_droprate=0., hidden_droprate=0., batchnorm=False):
		super(GRAND, self).__init__()
		self.in_dim = in_dim
		self.hid_dim = hid_dim
		self.S = S
		self.K = K
		self.n_class = n_class
		
		self.mlp = MLP(in_dim, hid_dim, n_class, input_droprate, hidden_droprate, batchmore)
		
		self.dropout = node_dropout
		self.node_dropout = nn.Dropout(node_dropout)
		
	def forward(self, graph, feats, training=True):		# Note that the output is applied by log_softmax.
		if training:
			output_list = []	
			for s in range(self.S):
				drop_feats = drop_node(feats, self.dropout)
				feat = GRANDConv(graph, drop_feats, self.K)
				output_list.append(th.log_softmax(self.mlp(feat), dim=-1))
			return output_list
		else:
			feat =  GRANDConv(graph, feats, self.K)
			return th.log_softmax(self.mlp(feat), dim = -1)

def consis_loss(log_ps_list, temp):	# log_ps: log probabilities
	ps = [th.exp(p) for p in log_ps_list]
	ps = th.stack(ps, dim=2)		# shape: (n_nodes, n_classes, S)
	Z_bar = th.mean(ps, dim=2)
	Z_bar_temp = th.pow(Z_bar, 1./temp)
	Z_bar_prime = (Z_bar_temp / th.sum(Z_bar_temp, dim=1, keepdim=True)).detach().unsqueeze(2)	# shape: (n_nodes, n_classes, 1)
	loss = th.mean(th.sum(th.pow(ps - Z_bar_prime, 2), dim=[0, 1]))
	return loss


def argument():
	parser = argparse.ArgumentParser(description='GRAND')
	
	parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset')
	parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU')
	parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
	parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping')
	parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 reg')
	parser.add_argument('--hid_dim', type=int, default=32, help='Hidden layer dimension')
	parser.add_argument('--dropnode_rate', type=float, default=0.5, help='Dropnode rate')
	parser.add_argument('--input_droprate', type=float, default=0.0, help='dropout rate of input layer')
	parser.add_argument('--hidden_droprate', type=float, default=0.0, help='dropout rate of hidden layer')
	parser.add_argument('--order', type=int, default=8, help='Propagation step')
	parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
	parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
	parser.add_argument('--lam', type=float, default=1., help='Coefficient of consistency regularization')
	parser.add_argument('--user_bn', action='store_true', default=False, help='Using Batch Normalization')
	
	args = parser.parse_args()
	
	if args.gpu != -1 and th.cuda.is_available():
		args.device = 'cuda:{}'.format(args.gpu)
	else:
		args.device = 'cpu'
		
	return args

	
if __name__ == '__main__':
	args = argument()
	print(args)
	
	

	
