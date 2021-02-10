import torch
import numpy as np
from neurodiffeq.conditions import BaseCondition, NoCondition

# I'm not supposed to hard code the coefficients: Bite me XO
sine_coeffs = np.array([
    +2.7841639984158566e+00, 0.0000000000000000e+00,
    -7.7819749512124448e-01, 0.0000000000000000e+00,
    -1.3050768742577226e-01, 0.0000000000000000e+00,
    -4.9015849881143721e-02, 0.0000000000000000e+00,
    -2.4522648837852077e-02, 0.0000000000000000e+00,
    -1.4309085187705703e-02, 0.0000000000000000e+00,
    -9.2002184434137185e-03, 0.0000000000000000e+00,
    -6.3257955704360885e-03, 0.0000000000000000e+00,
    -4.5689378172160252e-03, 0.0000000000000000e+00,
    -3.4268636052348397e-03, 0.0000000000000000e+00,
    -2.6481204272186891e-03, 0.0000000000000000e+00,
    -2.0964814943198605e-03, 0.0000000000000000e+00,
    -1.6933446413717327e-03, 0.0000000000000000e+00,
    -1.3909826639937207e-03, 0.0000000000000000e+00,
    -1.1591671649051640e-03, 0.0000000000000000e+00,
    -9.7805605112462270e-04, 0.0000000000000000e+00,
    -8.3423099630988370e-04, 0.0000000000000000e+00,
    -7.1837041729838397e-04, 0.0000000000000000e+00,
    -6.2385156406174699e-04], dtype=np.float64)

zero_coeffs = np.zeros_like(sine_coeffs)


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


class ConditionFactory:
    conditions = {
        'shift': SphericalShiftDirichletBVP,
        'normal': SphericalDirichletBVP,
        'reverse': ReverseBVP,
    }

    @staticmethod
    def get_condition(key, root_cfg, cfg):
        degrees = root_cfg.pde.degrees._list
        cls = ConditionFactory.conditions.get(cfg.type)
        if not cls:
            raise ValueError(f'Unknown condition type in {cfg}')

        if key in ['ur', 'utheta']:
            return cls(
                r0=root_cfg.pde.r0,
                R0=torch.tensor(zero_coeffs[degrees]),
                r1=root_cfg.pde.r1,
                R1=torch.tensor(zero_coeffs[degrees]),
            )
        elif key == 'uphi':
            return cls(
                r0=root_cfg.pde.r0,
                R0=root_cfg.pde.omega0 * root_cfg.pde.r0 * torch.tensor(sine_coeffs[degrees]),
                r1=root_cfg.pde.r1,
                R1=root_cfg.pde.omega1 * root_cfg.pde.r1 * torch.tensor(sine_coeffs[degrees]),
            )
        elif key == 'p':
            return cls(
                r0=root_cfg.pde.r0,
                R0=torch.tensor(zero_coeffs[degrees]),
            )
        else:
            raise ValueError(f'Unknown function {key}')

    @staticmethod
    def from_config(root_cfg):
        return {
            k: ConditionFactory.get_condition(k, root_cfg, cfg)
            for k, cfg in root_cfg.condition.items()
        }
