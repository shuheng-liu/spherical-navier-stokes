import yaml
import pytest
import shutil
from pathlib import Path
from config import Config


@pytest.fixture
def d1():
    return dict(
        a=dict(a=1, b=2),
        b=3,
    )


@pytest.fixture
def d2():
    return dict(
        a=dict(b=4, c=5),
        c=dict(e=6, f=7),
    )


@pytest.fixture
def d3():
    return {
        'a': 1,
        'b': [
            {'c': 2},
            [
                {'d': 3},
                {'e': {}},
            ]
        ]
    }


@pytest.fixture
def tmp_dir():
    path = Path('./tmp-test')
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.is_dir() and path.exists():
        shutil.rmtree(path)


@pytest.fixture
def default_config():
    return Config.from_yml_file('../default-config.yaml')


def test_config_list(d3):
    config = Config.auto_convert(d3)
    assert isinstance(config, Config)
    assert config._name == 'config'
    assert config.a == 1
    assert isinstance(config.b, Config)
    assert config.b._name == 'config.b'
    assert isinstance(config.b[0], Config)
    assert config.b[0]._name == 'config.b.0'
    assert config.b[0].c == 2
    assert isinstance(config.b[1], Config)
    assert config.b[1]._name == 'config.b.1'
    assert isinstance(config.b[1][0], Config)
    assert config.b[1][0]._name == 'config.b.1.0'
    assert config.b[1][0].d == 3
    assert isinstance(config.b[1][1], Config)
    assert config.b[1][1]._name == 'config.b.1.1'
    assert isinstance(config.b[1][1].e, Config)


def test_config(d1, d2):
    config = Config.auto_convert(d1)
    assert config._name == 'config'
    assert config.is_dict and not config.is_list
    assert isinstance(config.a, Config)
    assert config.a._name == 'config.a'
    assert config.a.is_dict and not config.a.is_list
    assert config.a.a == 1
    assert config.a.b == 2
    assert config.b == 3
    assert set(config) == {'a', 'b'}
    for k, v in config.items():
        assert isinstance(k, str)
        if k in ['a']:
            assert isinstance(v, Config)
        else:
            assert isinstance(v, int)

    config.update(d2)
    assert config._name == 'config'
    assert config.is_dict and not config.is_list
    assert isinstance(config.a, Config)
    assert config.a._name == 'config.a'
    assert config.a.is_dict and not config.a.is_list
    assert config.a.a == 1
    assert config.a.b == 4
    assert config.a.c == 5
    assert config.b == 3
    assert config.c._name == 'config.c'
    assert config.c.is_dict and not config.c.is_list
    assert isinstance(config.c, Config)
    assert config.c.e == 6
    assert config.c.f == 7
    assert set(config) == {'a', 'b', 'c'}

    for k, v in config.items():
        assert isinstance(k, str)
        if k in ['a', 'c']:
            assert isinstance(v, Config)
        else:
            assert isinstance(v, int)


def test_config_to_builtin(default_config):
    assert isinstance(Config.to_builtin(default_config), dict)


def test_config_to_yaml(default_config, tmp_dir):
    yml_str = default_config.to_yml()
    yaml.safe_load(yml_str)
    f = tmp_dir / 'test.yaml'
    default_config.to_yml_file(f)
    with open(f) as f:
        yaml.safe_load(f)
