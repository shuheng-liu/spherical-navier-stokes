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
from copy import deepcopy


@pytest.fixture
def root_config():
    return Config.from_yml_file('../default-config.yaml')


@pytest.fixture
def s(root_config):
    return Session(root_config)


@pytest.fixture
def modified_config(root_config):
    small_net = dict(
        module_type='fcnn',
        args=[1, 1],
        kwargs=dict(
            hidden_units=(32,),
            actv='swish'
        )
    )
    config = deepcopy(root_config)
    config.update(dict(
        pde=dict(
            degrees=[0]
        ),
        network=dict(
            ur=small_net,
            utheta=small_net,
            uphi=small_net,
            p=small_net,
        ),
        curriculum=dict(
            base_size=32,
            min_size=1,
            max_size=32,
            epochs_per_curriculum=1,
            n_curricula=2,
        ),
    ))
    config.__dict__['monitor'] = {}
    return config


@pytest.fixture
def s2(modified_config):
    return Session(modified_config)


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


def test_solver(modified_config, s2):
    assert isinstance(s2.solver, BaseSolver)
    weights_before = [p.detach().cpu().numpy() for n in s2.nets for p in n.parameters()]
    w_before = np.concatenate([w.flatten() for w in weights_before])
    s2.solver.fit(max_epochs=1)
    weights_after = [p.detach().cpu().numpy() for n in s2.nets for p in n.parameters()]
    w_after = np.concatenate([w.flatten() for w in weights_after])
    assert (w_before != w_after).any()


def test_fit(modified_config, s2):
    s2.fit()
    assert s2.solver.global_epoch \
           == modified_config.curriculum.n_curricula \
           * modified_config.curriculum.epochs_per_curriculum
