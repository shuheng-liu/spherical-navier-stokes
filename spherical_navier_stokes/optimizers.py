import torch
from torch import optim


class OptimizerFactory:
    optimizers = {
        'adam': optim.Adam,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adamax': optim.Adamax,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'sparseadam': optim.SparseAdam,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
    }

    @staticmethod
    def from_config(cfg):
        OptimizerClass = OptimizerFactory.optimizers.get(cfg.type.lower())
        if not OptimizerFactory:
            raise ValueError(f"Unknown optimizer class {cfg.type.lower()}")

        args = list(cfg.args) if cfg.args else []
        kwargs = {k: cfg.kwargs.__dict__[k] for k in cfg.kwargs} if cfg.kwargs else {}

        def optimizer_getter(params):
            return OptimizerClass(params, *args, **kwargs)

        return optimizer_getter
