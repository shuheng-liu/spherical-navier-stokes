import functools


def partial_class(cls, *args, **kwargs):
    class NewClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewClass
