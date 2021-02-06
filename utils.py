import functools
from pathlib import Path


def partial_class(cls, *args, **kwargs):
    class NewClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewClass


def safe_makedir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def safe_touch(fname, mode=664):
    Path(fname).touch(mode=664, exist_ok=True)