import osqp
import numpy as np
from scipy import sparse
import pdb
import torch

def CVaR(val,p,gamma,sign = -1):
	N = val.shape[0]
	P = sparse.csc_matrix(np.eye(N))
	q = sign*val
	A = np.vstack((np.eye(N),np.ones([1,N])))
	A = sparse.csc_matrix(A)
	l = np.append(np.zeros(N),1)
	u = np.append(np.minimum(p/gamma,np.ones(N)),1)

	# Create an OSQP object
	prob = osqp.OSQP()

	# Setup workspace and change alpha parameter
	prob.setup(P, q, A, l, u, alpha=1.0,verbose=False, polish=True)

	# Solve problem
	res = prob.solve()
	return q@res.x


def CVaR_weight(val,p,gamma,sign = -1,end_idx=None):
	q = torch.clamp(p/gamma,max=1.0)
	if end_idx is None:
		end_idx.p.shape[0]
	assert((p[end_idx:]==0).all())
	remain = 1.0
	if sign==1:
		idx = torch.argsort(val[0:end_idx])
	else:
		idx = torch.argsort(val[0:end_idx],descending=True)
	i = 0
	for i in range(end_idx):
		if q[idx[i]]>remain:
			q[idx[i]]=remain
			remain = 0.0
		else:
			remain-=q[idx[i]]
	return q


	