import pytest
import torch
from torch import sin, cos
import numpy as np
import functools
from math import pi as PI
from neurodiffeq import diff
from neurodiffeq.networks import FCNN, Swish
from neurodiffeq.generators import Generator3D, SamplerGenerator
from neurodiffeq.function_basis import ZonalSphericalHarmonics, ZonalSphericalHarmonicsLaplacian
from equations import ZonalHarmonicsNS


def pde_system(R_ur, R_utheta, R_uphi, R_p, r, theta, phi, mu=1.0, rho=1.0, harmonics_fn=None,
               harmonics_laplacian=None):
    # compute u_r, u_theta, u_phi and p from R
    basis = harmonics_fn(theta, phi)
    u_r = torch.sum(R_ur * basis, dim=(1,), keepdim=True)
    u_theta = torch.sum(R_utheta * basis, dim=(1,), keepdim=True)
    u_phi = torch.sum(R_uphi * basis, dim=(1,), keepdim=True)
    p = torch.sum(R_p * basis, dim=(1,), keepdim=True)

    def D(u):
        return u_r * diff(u, r) + (u_theta / r) * diff(u, theta)

    r_nonlinear = rho * (D(u_r) - (u_theta ** 2 + u_phi ** 2) / r)
    r_linear = \
        - diff(p, r) \
        + mu * (
                harmonics_laplacian(R_ur, r, theta, phi)
                - 2 * u_r / r ** 2
                - 2 * diff(u_theta, theta) / r ** 2
                - 2 * u_theta * cos(theta) / (sin(theta) * r ** 2)
        )
    f_r = r_nonlinear - r_linear

    theta_nonlinear = rho * (D(u_theta) + (u_theta * u_r - u_phi ** 2 * cos(theta) / sin(theta)) / r)
    theta_linear = \
        - diff(p, theta) / r \
        + mu * (
                harmonics_laplacian(R_utheta, r, theta, phi)
                + 2 / r ** 2 * diff(u_r, theta)
                - u_theta / (r ** 2 * sin(theta) ** 2)
        )
    f_theta = theta_nonlinear - theta_linear

    phi_nonlinear = rho * (D(u_phi) + (u_phi * u_r + u_theta * u_phi * cos(theta) / sin(theta)) / r)
    phi_linear = \
        mu * (
                harmonics_laplacian(R_uphi, r, theta, phi)
                - u_phi / (r ** 2 * sin(theta) ** 2)
        )
    f_phi = phi_nonlinear - phi_linear

    div_r = diff(u_r * r ** 2, r) / r ** 2
    div_theta = diff(u_theta * sin(theta), theta) / (r * sin(theta))
    div = div_r + div_theta

    return [f_r, f_theta, f_phi, div]


def test_zonal_harmonics_ns():
    assert ZonalHarmonicsNS.n_inputs == 3
    assert ZonalHarmonicsNS.n_outputs == 4

    r0, r1 = 0.1, 10.0
    omega0, omega1 = 10.0, 10.0
    generator = SamplerGenerator(Generator3D(grid=(5, 5, 5), xyz_min=(r0, 0.1, 0), xyz_max=(r1, PI - 0.1, PI * 2)))
    rho = np.random.rand() * 2 + 1
    mu = np.random.rand() * 2 + 1

    degrees = [0, 2, 4, 6, 8, 10]
    harmonics_fn = ZonalSphericalHarmonics(degrees=degrees)
    harmonics_laplacian = ZonalSphericalHarmonicsLaplacian(degrees=degrees)

    eq_res = ZonalHarmonicsNS(rho=rho, mu=mu, omega0=omega0, omega1=omega1, r0=r0, r1=r1, harmonics_fn=harmonics_fn)
    eq_exp = functools.partial(pde_system,
                               mu=mu, rho=rho, harmonics_fn=harmonics_fn, harmonics_laplacian=harmonics_laplacian)

    r, th, ph = generator.get_examples()

    nets = [FCNN(n_input_units=1, n_output_units=len(degrees), actv=Swish) for _ in range(4)]
    Rs = [net(r) for net in nets]

    res = eq_res(*Rs, r, th, ph)
    exp = eq_exp(*Rs, r, th, ph)

    for i, (r, e) in enumerate(zip(res, exp)):
        assert torch.isclose(r, e).all(), f'{i}th equation is not working well'
