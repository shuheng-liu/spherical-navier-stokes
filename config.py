from utils import recursive_update
import functools
import collections.abc
import yaml


class Config:
    def __init__(self, config_dict, name='config'):
        self._name = name
        self.__dict__.update(config_dict)

    def update(self, new_dict):
        self.__dict__ = recursive_update(self.__dict__, new_dict)

    @staticmethod
    def from_yml_file(path):
        with open(path) as f:
            return Config(yaml.safe_load(f))

    @staticmethod
    def from_yml(yml):
        return Config(yaml.safe_load(yml))

    def __getattribute__(self, item):
        if item.startswith('__') and item.endswith('__'):
            return object.__getattribute__(self, item)

        try:
            # Get a `method` object of the current instance
            # Don't mess with it unless you know what you are doing
            return functools.partial(getattr(Config, item), self)
        except AttributeError:
            v = self.__dict__.get(item, None)
            if isinstance(v, collections.abc.Mapping):
                return Config(v, name=item)
            return v

    def __repr__(self):
        return f'Config({self._name}){[k for k in self.__dict__]}'
