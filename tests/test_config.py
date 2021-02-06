from config import Config


def test_config():
    d1 = dict(
        a=dict(a=1, b=2),
        b=3,
    )
    config = Config(d1)
    assert config._name == 'config'
    assert isinstance(config.a, Config)
    assert config.a._name == 'a'
    assert config.a.a == 1
    assert config.a.b == 2
    assert config.b == 3

    d2 = dict(
        a=dict(b=4, c=5),
        c=dict(e=6, f=7),
    )
    config.update(d2)
    assert config._name == 'config'
    assert isinstance(config.a, Config)
    assert config.a._name == 'a'
    assert config.a.a == 1
    assert config.a.b == 4
    assert config.a.c == 5
    assert config.b == 3
    assert config.c._name == 'c'
    assert config.c.e == 6
    assert config.c.f == 7
