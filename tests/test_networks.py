import pytest
import torch
import torch.nn as nn
from spherical_navier_stokes.networks import ModelFactory
from neurodiffeq.networks import MonomialNN
from spherical_navier_stokes.config import Config


@pytest.fixture
def linear1():
    return dict(
        module_type='linear',
        args=[3, 4],
        kwargs=dict(
            bias=False,
        )
    )


@pytest.fixture
def linear2():
    return dict(
        module_type='linear',
        kwargs=dict(
            in_features=4,
            out_features=5,
            bias=True,
        )
    )


@pytest.fixture
def sequential(linear1, linear2):
    return dict(
        module_type='sequential',
        args=[
            linear1,
            dict(module_type='swish2'),
            linear2,
        ],
    )


@pytest.fixture
def monomial():
    return dict(
        module_type='monomialnn',
        kwargs=dict(
            degrees=[1, 2]
        )
    )


def test_model_factory(linear1, sequential, monomial):
    model = ModelFactory.from_config(
        Config.auto_convert(linear1)
    )
    assert isinstance(model, nn.Linear)
    assert model.in_features == linear1['args'][0]
    assert model.out_features == linear1['args'][1]
    assert model.bias is None

    model = ModelFactory.from_config(
        Config.auto_convert(sequential)
    )
    assert isinstance(model, nn.Sequential)
    x = torch.rand(10, 3)
    assert tuple(model(x).shape) == (10, 5)

    model = ModelFactory.from_config(
        Config.auto_convert(monomial)
    )
    assert isinstance(model, MonomialNN)
    assert set(model.degrees) == set(monomial['kwargs']['degrees'])
