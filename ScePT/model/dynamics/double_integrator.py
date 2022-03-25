import torch
from model.dynamics import Dynamic
from utils import block_diag
import numpy as np

class DoubleIntegrator(Dynamic):
    def init_constants(self):
        self.F = torch.eye(4, device=self.device, dtype=torch.float32)
        self.F[0:2, 2:] = torch.eye(2, device=self.device, dtype=torch.float32) * self.dt
        self.F_t = self.F.transpose(-2, -1)

    def integrate_samples(self, v, x=None):
        """
        Integrates deterministic samples of velocity.

        :param v: Velocity samples
        :param x: Not used for SI.
        :return: Position samples
        """
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        return torch.cumsum(v, dim=2) * self.dt + p_0

    def fast_integrate(self,x,u,dt):
        if isinstance(x,np.ndarray):
            xn = np.hstack(((x[...,2:4]+0.5*u*dt)*dt+x[...,0:2],x[...,2:4]+u*dt))
        elif isinstance(x,torch.Tensor):
            xn = torch.clone(x)
            xn[...,0:2]+=(x[...,2:4]+0.5*u*dt)*dt
            xn[...,2:4]+=u*dt
        else:
            raise NotImplementedError
        return xn
    def inverse_dyn(self,x,xp,dt):
        return (xp[...,2:]-x[...,2:])/dt

    def integrate_distribution(self, v_dist, x=None):
    	raise NotImplementedError

    def __call__(self,x,u,dt):
    	return self.fast_integrate(x,u,dt)
