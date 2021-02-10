import os
from spherical_navier_stokes.utils import recursive_update
import functools
import collections.abc
import yaml
from pathlib import Path


class Config:
    def __init__(self, cfg, name='config'):
        self._name = name
        if isinstance(cfg, collections.abc.Mapping):
            self.__dict__.update(cfg)
        elif isinstance(cfg, collections.abc.Sequence) and not isinstance(cfg, str):
            self.__dict__.update({'_list': cfg})
        else:
            raise TypeError(f'Illegal type(cfg) = {type(cfg)}')

    def update(self, new_dict):
        self.__dict__ = recursive_update(self.__dict__, new_dict)

    @staticmethod
    def auto_convert(data, name='config', base_name=None):
        if base_name is not None:
            name = f'{base_name}.{name}'
        if isinstance(data, Config):
            return data
        elif isinstance(data, collections.abc.Mapping):
            return Config(data, name=name)
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            return Config(data, name=name)
        return data

    @staticmethod
    def from_yml_file(path, name=None):
        with open(path) as f:
            return Config.auto_convert(yaml.safe_load(f), name=name or os.path.basename(path).replace('.', '_'))

    @staticmethod
    def from_yml(yml, name=None):
        return Config.auto_convert(yaml.safe_load(yml), name=name or 'root')

    # Don't mess with this method unless you know what you are doing
    def __getattribute__(self, item):
        if item.startswith('__') and item.endswith('__'):
            return object.__getattribute__(self, item)

        if item in ['_list', '_name']:
            return self.__dict__[item]

        if item == 'is_list':
            return '_list' in self.__dict__
        elif item == 'is_dict':
            return '_list' not in self.__dict__

        try:
            # Get a `method` object of the current instance
            return functools.partial(getattr(Config, item), self)
        except AttributeError:
            return Config.auto_convert(self.__dict__.get(item, None), name=item, base_name=self._name)

    def __repr__(self):
        if self.is_dict:
            return f'Config({self._name}){[k for k in self.__dict__ if not k.startswith("_")]}'
        else:
            return f'Config({self._name})[0..{len(self._list)})'

    def __iter__(self):
        if self.is_list:
            for item in self.__dict__['_list']:
                yield item
        else:
            for k in self.__dict__:
                if not k.startswith('_name'):
                    yield k

    def __getitem__(self, item):
        if isinstance(item, int):
            return Config.auto_convert(self._list[item], name=item, base_name=self._name)
        else:
            return Config.auto_convert(self.__dict__[item], name=item, base_name=self._name)

    def items(self):
        if self.is_list:
            for i, item in enumerate(self.__dict__['_list']):
                yield Config.auto_convert(item, name=item, base_name=self._name)
        else:
            for k in self.__dict__:
                if not k.startswith('_name'):
                    yield k, Config.auto_convert(self.__dict__[k], name=k, base_name=self._name)

    @staticmethod
    def to_builtin(config):
        if not isinstance(config, Config):
            return config

        if config.is_dict:
            return {k: Config.to_builtin(config.__dict__[k]) for k in config}

        if config.is_list:
            return [Config.to_builtin(item) for item in Config.__dict__['_list']]

    def to_yml(self):
        return yaml.safe_dump(Config.to_builtin(self))

    def to_yml_file(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(Config.to_builtin(self), stream=f)


try:
    default_config = Config.from_yml_file(Path(__file__).parent / 'default-config.yaml', name='DEFAULT-CONFIG')
except FileNotFoundError:
    default_config = None
