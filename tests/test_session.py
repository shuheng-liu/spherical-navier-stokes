import pytest
import torch
import torch.nn as nn
import numpy as np
from neurodiffeq.callbacks import MonitorCallback
from neurodiffeq.conditions import BaseCondition
from neurodiffeq.solvers import BaseSolver
from session import Session
from weighting import ScalarComposition, get_fn_by_name, SoftStep
from config import Config
from curriculum import BaseCurriculumLearner


@pytest.fixture
def root_config():
    return Config.from_yml_file('../default-config.yaml')


@pytest.fixture
def s(root_config):
    return Session(root_config)


def test_weighting(root_config, s):
    for eq, w_fn in s.weight_fns.items():
        assert isinstance(w_fn, ScalarComposition)
        assert w_fn.alpha == getattr(root_config.weighting, eq).weight
        assert isinstance(w_fn.fn, get_fn_by_name(getattr(root_config.weighting, eq).type))
        if isinstance(w_fn.fn, SoftStep):
            for arg_name, arg_value in getattr(root_config.weighting, eq).args.items():
                assert getattr(w_fn.fn, arg_name) == arg_value


def test_equations(root_config, s):
    assert s.pdes.r0 == root_config.pde.r0
    assert s.pdes.r1 == root_config.pde.r1
    assert s.pdes.omega0 == root_config.pde.omega0
    assert s.pdes.omega1 == root_config.pde.omega1
    assert s.pdes.rho == root_config.pde.rho
    assert s.pdes.mu == root_config.pde.mu
    assert s.harmonics_fn.degrees == list(root_config.pde.degrees.items())


def test_networks(root_config, s):
    for net in s.nets:
        assert isinstance(net, nn.Module)


def test_optimizer(root_config, s):
    assert isinstance(s.optimizer, torch.optim.Optimizer)


def test_monitors(root_config, s):
    assert isinstance(s.monitor_callbacks, list)
    for m in s.monitor_callbacks:
        isinstance(m, MonitorCallback)


def test_curriculum(root_config, s):
    assert isinstance(s.curriculum, BaseCurriculumLearner)


def test_conditions(root_config, s):
    for c in s.conditions:
        assert isinstance(c, BaseCondition)


def test_solver(root_config, s):
    assert isinstance(s.solver, BaseSolver)
    weights_before = [p.detach().cpu().numpy() for n in s.nets for p in n.parameters()]
    w_before = np.concatenate([w.flatten() for w in weights_before])
    s.solver.fit(max_epochs=1)
    weights_after = [p.detach().cpu().numpy() for n in s.nets for p in n.parameters()]
    w_after = np.concatenate([w.flatten() for w in weights_after])
    assert (w_before != w_after).any()
