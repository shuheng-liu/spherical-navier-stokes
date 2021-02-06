from utils import recursive_update
import functools
import collections.abc
import yaml


def auto_convert_to_config(data, name='config'):
    if isinstance(data, Config):
        return data
    if isinstance(data, collections.abc.Mapping):
        return Config(data, name=name)
    return data


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
            return auto_convert_to_config(self.__dict__.get(item, None), name=item)

    def __repr__(self):
        return f'Config({self._name}){[k for k in self.__dict__]}'

    def __iter__(self):
        for k in self.__dict__:
            if k != '_name':
                yield k

    def items(self):
        for k in self.__dict__:
            if k != '_name':
                yield k, auto_convert_to_config(self.__dict__[k])
