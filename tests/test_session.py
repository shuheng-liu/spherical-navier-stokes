import pytest
import torch.nn as nn
from session import Session
from weighting import ScalarComposition, get_fn_by_name, SoftStep
from config import Config


@pytest.fixture
def root_config():
    return Config.from_yml_file('../default-config.yaml')


@pytest.fixture
def s():
    return Session()


def test_set_weighting(root_config, s):
    s.set_weighting(root_config)
    for eq, w_fn in s.weight_fns.items():
        assert isinstance(w_fn, ScalarComposition)
        assert w_fn.alpha == getattr(root_config.weighting, eq).weight
        assert isinstance(w_fn.fn, get_fn_by_name(getattr(root_config.weighting, eq).type))
        if isinstance(w_fn.fn, SoftStep):
            for arg_name, arg_value in getattr(root_config.weighting, eq).args.items():
                assert getattr(w_fn.fn, arg_name) == arg_value


def test_set_equations(root_config, s):
    s.set_equations(root_config)
    assert s.pdes.r0 == root_config.pde.r0
    assert s.pdes.r1 == root_config.pde.r1
    assert s.pdes.omega0 == root_config.pde.omega0
    assert s.pdes.omega1 == root_config.pde.omega1
    assert s.pdes.rho == root_config.pde.rho
    assert s.pdes.mu == root_config.pde.mu
    assert s.pdes.harmonics_fn.degrees == list(root_config.pde.degrees.items())


def test_set_networks(root_config, s):
    s.set_networks(root_config)
    for net in s.nets:
        assert isinstance(net, nn.Module)
