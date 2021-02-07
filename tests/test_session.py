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
    wconfig = root_config.weighting

    s.set_weighting(weighting_cfg=wconfig)
    for eq, w_fn in s.weight_fns.items():
        assert isinstance(w_fn, ScalarComposition)
        assert w_fn.alpha == getattr(wconfig, eq).weight
        assert isinstance(w_fn.fn, get_fn_by_name(getattr(wconfig, eq).type))
        if isinstance(w_fn.fn, SoftStep):
            for arg_name, arg_value in getattr(wconfig, eq).args.items():
                assert getattr(w_fn.fn, arg_name) == arg_value


def test_set_equations(root_config, s):
    pde_cfg = root_config.pde
    numerical_cfg = root_config.numerical

    s.set_equations(pde_cfg=pde_cfg, numerical_cfg=numerical_cfg)
    assert s.pdes.r0 == pde_cfg.r0
    assert s.pdes.r1 == pde_cfg.r1
    assert s.pdes.omega0 == pde_cfg.omega0
    assert s.pdes.omega1 == pde_cfg.omega1
    assert s.pdes.rho == pde_cfg.rho
    assert s.pdes.mu == pde_cfg.mu
    assert s.pdes.harmonics_fn.degrees == numerical_cfg.degrees


def test_set_networks(root_config, s):
    s = Session()
    s.set_networks(root_config.network)
    for net in s.nets:
        assert isinstance(net, nn.Module)
