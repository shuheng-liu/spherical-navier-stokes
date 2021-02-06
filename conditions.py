import torch
from neurodiffeq.conditions import BaseCondition


class SphericalDirichletBVP(BaseCondition):
    def __init__(self, r0, R0, r1, R1, normalizer=None):
        super(SphericalDirichletBVP, self).__init__()
        if r0 > r1:
            raise ValueError(f"r0 = {r0} > r1 = {r1}")
        self.r0, self.r1 = r0, r1
        self.R0, self.R1 = R0, R1
        self.r_dist = r1 - r0
        self.normalizer = normalizer or (r1 - r0)

    def parameterize(self, output_tensor, r):
        dr0 = (r - self.r0) / self.r_dist
        dr1 = (self.r1 - r) / self.r_dist
        exponent = - (dr1 * dr0) / self.normalizer

        return self.R0 * dr1 + self.R1 * dr0 + (1 - torch.exp(exponent)) * output_tensor


class ReverseBVP(BaseCondition):
    def __init__(self, r0, R0, normalizer=1.0):
        super(ReverseBVP, self).__init__()
        self.r0 = r0
        self.R0 = R0
        self.normalizer = normalizer

    def parameterize(self, output_tensor, r):
        return (1 - torch.exp(- torch.abs(r - self.r0) / self.normalizer)) * output_tensor + self.R0


class SphericalShiftDirichletBVP(BaseCondition):
    def __init__(self, r0, R0, r1, R1):
        super(SphericalShiftDirichletBVP, self).__init__()
        self.r0, self.r1 = r0, r1
        self.R0, self.R1 = R0, R1
        self.dr = r1 - r0
        self.r0_tensor = torch.tensor([[r0]])  # no need for gradient
        self.r1_tensor = torch.tensor([[r1]])

    def enforce(self, net, r):
        nr = net(torch.cat([r], dim=1))
        n0 = net(self.r0_tensor)
        n1 = net(self.r1_tensor)
        return nr - ((n1 - n0 + self.R0 - self.R1) / self.dr * (r - self.r0) + (n0 - self.R0))
