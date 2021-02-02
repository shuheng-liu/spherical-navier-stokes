import torch
from neurodiffeq.conditions import BaseCondition


class SphericalDirichletBVP(BaseCondition):
    def __init__(self, r_0, R_0, r_1, R_1, normalizer=None):
        super(SphericalDirichletBVP, self).__init__()
        if r_0 > r_1:
            raise ValueError(f"r_0 = {r_0} > r_1 = {r_1}")
        self.r_0, self.r_1 = r_0, r_1
        self.R_0, self.R_1 = R_0, R_1
        self.r_dist = r_1 - r_0
        self.normalizer = normalizer or (r_1 - r_0)

    def parameterize(self, output_tensor, r):
        dr0 = (r - self.r_0) / self.r_dist
        dr1 = (self.r_1 - r) / self.r_dist
        exponent = - (dr1 * dr0) / self.normalizer

        return self.R_0 * dr1 + self.R_1 * dr0 + (1 - torch.exp(exponent)) * output_tensor


class ReverseBVP(BaseCondition):
    def __init__(self, r_0, R_0, normalizer=1.0):
        super(ReverseBVP, self).__init__()
        self.r_0 = r_0
        self.R_0 = R_0
        self.normalizer = normalizer

    def parameterize(self, output_tensor, r):
        return (1 - torch.exp(- torch.abs(r - self.r_0) / self.normalizer)) * output_tensor + self.R_0
