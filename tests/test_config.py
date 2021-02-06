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
    assert set(config) == {'a', 'b'}
    for k, v in config.items():
        assert isinstance(k, str)
        if k in ['a']:
            assert isinstance(v, Config)
        else:
            assert isinstance(v, int)

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
