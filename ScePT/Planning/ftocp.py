from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np

##### MY CODE ######
class FTOCP(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0 and terminal contraints
		- buildNonlinearProgram: builds the nonlinear program solved by the above solve methos
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t
	"""
	def __init__(self, N,M, dt,W,L):
		# Define variables
		self.N    = N
		self.dt   = dt
		self.n    = 4
		self.d    = 2
		self.M    = M
		self.xRef = None
		self.W    = W
		self.L    = L
		self.obs  = list()
		self.x_lb = [-np.inf,-np.inf,0,-2*np.pi]
		self.x_ub = [np.inf,np.inf,40,2*np.pi]

		self.u_lb = [-5.0,-0.1]
		self.u_ub = [5.0,0.1]



		self.feasible = 0
		self.xPredOld =[]
		self.yPredOld =[]

		self.solOld =[]
		self.xGuessTot = None


	def buildandsolve(self,nodes,x0_val, xdes,ypreds,w):
		# Define variables
		n = self.n
		d = self.d
		N = self.N
		M = self.M
		
		Nnodes = len(nodes)
		

		X     = SX.sym('X', n*(M*N+1))
		x0 = X[0:n]
		xbr = [None]*M
		
		U = SX.sym('U', d*(M*(N-1)+1))
		u0 = U[0:d]
		ubr = [None]*M 
		for i in range(M):
			xbr[i] = X[(i*N+1)*n:(i*N+N+1)*n].reshape((n,N)).T
			ubr[i] = U[(i*(N-1)+1)*d:((i+1)*(N-1)+1)*d].reshape((d,N-1)).T
		slack0 = SX.sym('s', N*M)
		slack = slack0.reshape((N,M)).T

		# Define dynamic constraints
		constraint = list()
		for i in range(M):
			constraint = vertcat(constraint,xbr[i][0,0]-x0[0]-self.dt*x0[2]*casadi.cos(x0[3]))
			constraint = vertcat(constraint,xbr[i][0,1]-x0[1]-self.dt*x0[2]*casadi.sin(x0[3]))
			constraint = vertcat(constraint,xbr[i][0,2]-x0[2]-self.dt*u0[0])
			constraint = vertcat(constraint,xbr[i][0,3]-x0[3]-self.dt*u0[1])

			for j in range(N-1):
				constraint = vertcat(constraint,xbr[i][j+1,0]-xbr[i][j,0]-self.dt*xbr[i][j,2]*casadi.cos(xbr[i][j,3]))
				constraint = vertcat(constraint,xbr[i][j+1,1]-xbr[i][j,1]-self.dt*xbr[i][j,2]*casadi.sin(xbr[i][j,3]))
				constraint = vertcat(constraint,xbr[i][j+1,2]-xbr[i][j,2]-self.dt*ubr[i][j,0])
				constraint = vertcat(constraint,xbr[i][j+1,3]-xbr[i][j,3]-self.dt*ubr[i][j,1])
		eq_count = constraint.shape[0]

		# Obstacle constraints

		if Nnodes>0:
			for i in range(M):
				for j in range(Nnodes):
					for k in range(N):
						constraint = vertcat(constraint, ( ( xbr[i][k,0] - ypreds[j][i][k,0] )**2/(self.L/1.414+nodes[j].length/1.414)**2 +
														   ( xbr[i][k,1] - ypreds[j][i][k,1] )**2/(self.W/1.414+nodes[j].width/1.414)**2 + slack[i,k] ) );

		ineq_count = constraint.shape[0]-eq_count

		# Defining Cost
		cost = 0
		cost_x   = 1.
		cost_y   = 5.
		cost_acc = 5.0
		cost_ste = 20.0
		cost_slack = 1e6
		cost_R = DM([cost_acc,cost_ste])
		cost_Q = DM([cost_x,cost_y])
		cost = sum1(u0**2*cost_R)
		for i in range(M):
			for k in range(N-1):

				cost+=(sum1((xbr[i][k,:2].T-xdes[k][:2])**2*cost_Q)+sum1(ubr[i][k,:].T**2*cost_R)+slack[i,k]*cost_slack)*w[i]

			cost+= (sum1((xbr[i][N-1,:2].T-xdes[N-1][:2])**2*cost_Q)+slack[i,N-1]*cost_slack)*w[i]



		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive"}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X,U, slack0), 'f':cost, 'g':constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force: 1) n*N state dynamics and 2) inequality constraints (set to zero as we have slack)
		self.lbg_dyanmics = [0]*eq_count + [1]*ineq_count
		self.ubg_dyanmics = [0]*eq_count + [10000]*ineq_count

		self.lbx = x0_val.tolist() + self.x_lb*(N*M) + self.u_lb*(M*(N-1)+1) + [0]*(N*M)
		self.ubx = x0_val.tolist() + self.x_ub*(N*M) + self.u_ub*(M*(N-1)+1) + [np.inf]*(N*M)
		
		if self.xGuessTot is not None and self.xGuessTot.shape[0]==nlp['x'].shape[0]:
			sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0 = self.xGuessTot)

		else:
			sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
		# sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
		# Check solution flag
		if (self.solver.stats()['success']):
			self.feasible = 1
		else:
			sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)



		# Store optimal solution
		x = np.array(sol["x"])
		self.xSol  = x[0:n*(M*N+1)].reshape((M*N+1,n))
		self.uSol  = x[n*(M*N+1):n*(M*N+1)+d*(M*(N-1)+1)].reshape((M*(N-1)+1,d))
		self.slack = x[n*(M*N+1)+d*(M*(N-1)+1):]

		self.xGuessTot = x

		# Check solution flag
		if (self.solver.stats()['success']):
			self.feasible = 1
		else:
			self.feasible = 0


