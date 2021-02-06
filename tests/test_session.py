import pytest
from session import Session
from weighting import ScalarComposition, get_fn_by_name, SoftStep
from config import Config


@pytest.fixture
def root_config():
    return Config.from_yml_file('../default-config.yaml')


def test_set_weighting(root_config):
    wconfig = root_config.weighting

    s = Session()
    s.set_weighting(weighting_cfg=wconfig)
    for eq, w_fn in s.weight_fns.items():
        assert isinstance(w_fn, ScalarComposition)
        assert w_fn.alpha == getattr(wconfig, eq).weight
        assert isinstance(w_fn.fn, get_fn_by_name(getattr(wconfig, eq).type))
        if isinstance(w_fn.fn, SoftStep):
            for arg_name, arg_value in getattr(wconfig, eq).args.items():
                assert getattr(w_fn.fn, arg_name) == arg_value
