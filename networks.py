import torch
from torch import nn
from neurodiffeq.networks import FCNN, MonomialNN, SinActv
from config import Config
from utils import partial_class


class SwishN(nn.Module):
    def __init__(self, order=1):
        super(SwishN, self).__init__()
        self.order = order

    def __call__(self, x):
        return torch.sigmoid(x) * (x ** self.order)


Swish0 = nn.Sigmoid
Swish1 = partial_class(SwishN, order=1)
Swish2 = partial_class(SwishN, order=2)
Swish3 = partial_class(SwishN, order=3)
Swish4 = partial_class(SwishN, order=4)
Swish5 = partial_class(SwishN, order=5)
Swish = Swish1


class ResBlock(nn.Module):
    def __init__(self, n_neurons, act1=None, act2=None):
        super(ResBlock, self).__init__()
        self.lin1 = nn.Linear(n_neurons, n_neurons)
        self.act1 = act1 or Swish()
        self.lin2 = nn.Linear(n_neurons, n_neurons)
        self.act2 = act2 or Swish()

    def forward(self, x):
        y = x
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.act2(x + y)


class Resnet(nn.Sequential):
    def __init__(self, n_input_units, n_output_units, n_res_blocks, n_res_units, actv=None):
        if actv is None:
            actv = Swish
        layers = []
        layers.append(nn.Linear(n_input_units, n_res_units))
        layers.append(actv())
        for i in range(n_res_blocks):
            layers.append(ResBlock(n_res_units, act1=actv(), act2=actv()))
        layers.append(nn.Linear(n_res_units, n_output_units))
        nn.Sequential.__init__(self, *layers)


class ModelFactory:
    activations = dict(
        sin=SinActv,
        tanh=nn.Tanh,
        sigmoid=nn.Sigmoid,
        hardswish=nn.Hardswish,
        relu=nn.ReLU,
        swish=Swish,
        swish1=Swish1,
        swish2=Swish2,
        swish3=Swish3,
        swish4=Swish4,
        swish5=Swish5,
    )

    models = dict(
        resnet=Resnet,
        fcnn=FCNN,
        monomialnn=MonomialNN,
        linear=nn.Linear,
        softmax=nn.Softmax,
        sequential=nn.Sequential,
    )
    models.update(activations)

    @staticmethod
    def get_model(cfg):
        model = ModelFactory.models.get(cfg.module_type.lower())
        if not model:
            raise ValueError(f"Unknown model type in {cfg}, must be one of {list(ModelFactory.models.keys())}")
        return model

    @staticmethod
    def get_activation(actv_name):
        if not isinstance(actv_name, str):
            raise TypeError(f"actv_name={actv_name} must be str, got {type(actv_name)}")
        actv = ModelFactory.activations.get(actv_name.lower())
        if not actv:
            raise ValueError(
                f"Unknown activation type {actv_name}, must be one of {list(ModelFactory.activations.keys())}"
            )
        return actv

    @staticmethod
    def parse_args(cfg):
        if cfg.args is None:
            return []
        if not isinstance(cfg.args, Config):
            raise TypeError(f"cfg.args={cfg.args} must be of type Config, not {type(cfg.args)}")
        if not cfg.args.is_list:
            raise ValueError(f"cfg.args must be a list config, got {cfg.args}")

        return [ModelFactory.from_config(arg) for arg in cfg.args.items()]

    @staticmethod
    def parse_kwargs(cfg):
        if cfg.kwargs is None:
            return {}
        if not isinstance(cfg.kwargs, Config):
            raise TypeError(f"cfg.kwargs={cfg.kwargs} must be of type Config, not {type(cfg.kwargs)}")
        if not cfg.kwargs.is_dict:
            raise ValueError(f"cfg.kwargs must be a dict config, got {cfg.kwargs}")

        return {
            k: ModelFactory.get_activation(v) if k == 'actv' else ModelFactory.from_config(v)
            for k, v in cfg.kwargs.items()
        }

    @staticmethod
    def from_config(cfg):
        if not isinstance(cfg, Config):
            return cfg

        if cfg.module_type is None:
            if cfg.is_list:
                return list(cfg)
            else:
                return {k: cfg.__dict__[k] for k in cfg}

        ModelClass = ModelFactory.get_model(cfg)
        args = ModelFactory.parse_args(cfg)
        kwargs = ModelFactory.parse_kwargs(cfg)
        return ModelClass(*args, **kwargs)
