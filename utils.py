import os
import dill
import datetime
import functools
import collections.abc
from pathlib import Path


def partial_class(cls, *args, **kwargs):
    class NewClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewClass


def safe_mkdir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def safe_touch(fname):
    Path(fname).touch(exist_ok=True)


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def dump(obj, path):
    safe_mkdir(os.path.dirname(path))
    with open(path, 'wb') as f:
        dill.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def timestr():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
