import torch
import numpy as np
from neurodiffeq.networks import FCNN
from neurodiffeq.generators import Generator1D, SamplerGenerator
from neurodiffeq.function_basis import ZonalSphericalHarmonics

from conditions import ReverseBVP, SphericalDirichletBVP, SphericalShiftDirichletBVP

N_SAMPLES = 100
degrees = [0, 2, 4, 6, 8]
harmonics_fn = ZonalSphericalHarmonics(degrees=degrees)


def test_reverse_bvp():
    net = FCNN(1, 1)
    r0 = np.random.rand()
    R0 = torch.rand(len(degrees))
    condition = ReverseBVP(r0, R0, normalizer=np.random.rand())
    r = r0 * torch.ones(N_SAMPLES, 1)
    assert torch.isclose(condition.enforce(net, r), R0).all()


def test_spherical_dirichlet_bvp():
    net = FCNN(1, 1)
    r0 = np.random.rand()
    r1 = np.random.rand() + r0
    R0 = torch.rand(len(degrees))
    R1 = torch.rand(len(degrees))
    condition = SphericalDirichletBVP(r0, R0, r1, R1, normalizer=np.random.rand())
    r = r0 * torch.ones(N_SAMPLES, 1)
    assert torch.isclose(condition.enforce(net, r), R0).all()
    r = r1 * torch.ones(N_SAMPLES, 1)
    assert torch.isclose(condition.enforce(net, r), R1).all()


def test_spherical_shift_dirichlet_bvp():
    net = FCNN(1, 1)
    r0 = np.random.rand()
    r1 = np.random.rand() + r0
    R0 = torch.rand(len(degrees))
    R1 = torch.rand(len(degrees))
    condition = SphericalShiftDirichletBVP(r0, R0, r1, R1)
    r = r0 * torch.ones(N_SAMPLES, 1)
    assert torch.isclose(condition.enforce(net, r), R0).all()
    r = r1 * torch.ones(N_SAMPLES, 1)
    assert torch.isclose(condition.enforce(net, r), R1).all()

