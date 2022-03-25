
class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, model_registrar, xz_size, node_type):
        self.dt = dt
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        self.model_registrar = model_registrar
        self.node_type = node_type
        self.init_constants()
        self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self):
        pass

    def create_graph(self, xz_size):
        pass

    def integrate_samples(self, s, x):
        raise NotImplementedError

    def integrate_distribution(self, dist, x):
        raise NotImplementedError
    def fast_integrate(self,x,u,dt):
        raise NotImplementedError

    def create_graph(self, xz_size):
        pass
    def forward(self,x,u,dt):
        return self.fast_integrate(x,u,dt)

    def inverse_dyn(self,x,xp,dt):
        raise NotImplementedError




