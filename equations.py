import torch
from abc import ABC, abstractmethod
from functools import partial
from neurodiffeq import diff
from neurodiffeq.operators import spherical_curl, spherical_grad, spherical_div, spherical_laplacian, \
    spherical_vector_laplacian
from neurodiffeq.function_basis import ZonalSphericalHarmonics, ZonalSphericalHarmonicsLaplacian


class EquationSystem(ABC):
    n_inputs = ...
    n_outputs = ...

    @abstractmethod
    def __call__(self, *args):
        pass


class BaseNavierStokes(EquationSystem):
    n_inputs = 3  # spatial coordinates
    n_outputs = 4  # velocity field + pressure

    def __init__(self, rho, mu, u_scale, l_scale):
        self.rho = rho
        self.mu = mu
        self.u_scale = u_scale
        self.l_scale = l_scale

    @property
    def reynolds_number(self):
        return self.rho * self.u_scale * self.l_scale / self.mu


class SphericalNS(BaseNavierStokes):
    def __init__(self, rho, mu, omega0, omega1, r0, r1):
        self.omega0, self.omega1 = omega0, omega1
        self.r0, self.r1 = r0, r1
        u_scale = max(omega0 * r0, omega1 * r1)
        l_scale = max(r0, r1)
        super(SphericalNS, self).__init__(rho=rho, mu=mu, u_scale=u_scale, l_scale=l_scale)


class ZonalHarmonicsNS(SphericalNS):
    def __init__(self, rho, mu, omega0, omega1, r0, r1, harmonics_fn):
        super(ZonalHarmonicsNS, self).__init__(rho=rho, mu=mu, omega0=omega0, omega1=omega1, r0=r0, r1=r1)
        self.harmonics_fn = harmonics_fn  # type: ZonalSphericalHarmonics

    def __call__(self, Rur, Ruth, Ruph, Rp, r, th, ph):
        basis = self.harmonics_fn(th, ph)
        sin = torch.sin(th)
        cot = torch.cos(th) / sin
        r2 = r ** 2
        r2sin = r2 * sin
        r2sin2 = r2sin * sin

        ur = (Rur * basis).sum(dim=1, keepdim=True)
        uth = (Ruth * basis).sum(dim=1, keepdim=True)
        uph = (Ruph * basis).sum(dim=1, keepdim=True)
        p = (Rp * basis).sum(dim=1, keepdim=True)

        dr = partial(diff, t=r)
        dth = partial(diff, t=th)
        ur_r, uth_r, uph_r = dr(ur), dr(uth), dr(uph)
        ur_th, uth_th, uph_th = dth(ur), dth(uth), dth(uph)

        r_nl = self.rho * (ur * ur_r + uth / r * ur_th - (uth ** 2 + uph ** 2) / r)
        r_l = -dr(p) + self.mu * (
                dr(r2 * ur_r) / r2 + dth(sin * ur_th) / r2sin  # laplacian
                - 2 / r2 * (ur + uth_th + uth * cot)
        )
        th_nl = self.rho * (ur * uth_r + uth / r * uth_th + (uth * ur - uph ** 2 * cot) / r)
        th_l = -dth(p) / r + self.mu * (
                dr(r2 * uth_r) / r2 + dth(sin * uth_th) / r2sin  # laplacian
                + 2 * ur_th / r2 - uth / r2sin2
        )
        ph_nl = self.rho * (ur * uph_r + uth / r * uph_th + (ur + uth * cot) * uph / r)
        ph_l = self.mu * (
                dr(r2 * uph_r) / r2 + dth(sin * uph_th) / r2sin  # laplacian
                - uph / r2sin2
        )

        div = dr(r2 * ur) / r2 + dth(uth * sin) / (r * sin)

        return [r_nl - r_l, th_nl - th_l, ph_nl - ph_l, div]
