import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from model.model_utils import to_one_hot, simplelinear
import pdb


class clique_gibbs_distr(nn.Module):
	def __init__(self,state_enc_dim,edge_encoding_dim,z_dim,edge_types,node_types,hyperparams,device,node_hidden_dim=[64,64],edge_hidden_dim = [64,64]):
		super(clique_gibbs_distr, self).__init__()
		self.edge_encoding_dim = edge_encoding_dim
		self.z_dim = z_dim
		self.state_enc_dim = state_enc_dim
		self.node_types = node_types
		self.edge_types = edge_types
		self.et_name=dict()
		for et in self.edge_types:
			self.et_name[et]=et[0].name+'->'+et[1].name
		self.device = device
		self.node_factor = nn.ModuleDict()
		self.edge_factor = nn.ModuleDict()
		for node_type in self.node_types:
			self.node_factor[node_type.name] = simplelinear(state_enc_dim[node_type],z_dim,device,node_hidden_dim).to(self.device)
		for edge_type in self.edge_types:
			self.edge_factor[self.et_name[edge_type]] = simplelinear(edge_encoding_dim[edge_type],z_dim*z_dim,device,edge_hidden_dim).to(self.device)
	def forward(self,node_types,node_encs,edge_encoding,clique_is_robot=None):
		if clique_is_robot is None:
			N = len(node_types)
			res = torch.zeros([self.z_dim]*N).to(self.device)
			for i in range(0,N):
				node_factor_i = self.node_factor[node_types[i].name](node_encs[i]).to(self.device)
				dim = [1]*N 
				dim[i]=self.z_dim
				repeats = [self.z_dim]*N 
				repeats[i]=1
				node_factor_i = node_factor_i.view(dim).repeat(repeats)
				res += node_factor_i
			for edge,enc in edge_encoding.items():
				i,j = edge
				if (node_types[i],node_types[j]) in self.edge_types:
					edge_factor_ij = self.edge_factor[node_types[i].name+'->'+node_types[j].name](enc).to(self.device)
					dim = [1]*N 
					dim[i]=self.z_dim
					dim[j]=self.z_dim
					repeats = [self.z_dim]*N 
					repeats[i]=1
					repeats[j]=1
					edge_factor_ij = edge_factor_ij.view(dim).repeat(repeats)
					res += edge_factor_ij
		else:
			N = len(node_types)-sum(clique_is_robot)
			if N==0:
				return torch.tensor([1.0]).to(self.device)
			adjusted_idx = list()
			idx = 0
			for i in range(len(node_types)):
				adjusted_idx.append(idx)
				idx+=(not clique_is_robot[i])

			res = torch.zeros([self.z_dim]*N).to(self.device)
			for i in range(0,len(node_types)):
				if not clique_is_robot[i]:
					node_factor_i = self.node_factor[node_types[i].name](node_encs[i]).to(self.device)
					dim = [1]*N 
					dim[adjusted_idx[i]]=self.z_dim
					repeats = [self.z_dim]*N 
					repeats[adjusted_idx[i]]=1
					node_factor_i = node_factor_i.view(dim).repeat(repeats)
					res += node_factor_i
			for edge,enc in edge_encoding.items():
				i,j = edge
				if not (clique_is_robot[i] or clique_is_robot[j]):
					if (node_types[i],node_types[j]) in self.edge_types:
						edge_factor_ij = self.edge_factor[node_types[i].name+'->'+node_types[j].name](enc).to(self.device)
						dim = [1]*N 
						dim[adjusted_idx[i]]=self.z_dim
						dim[adjusted_idx[j]]=self.z_dim
						repeats = [self.z_dim]*N 
						repeats[adjusted_idx[i]]=1
						repeats[adjusted_idx[j]]=1
						edge_factor_ij = edge_factor_ij.view(dim).repeat(repeats)
						res += edge_factor_ij
		return res




