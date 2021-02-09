from itertools import chain
from weighting import ScalarComposition, get_fn_by_name, WeightedResiduals
from equations import ZonalHarmonicsNS
from neurodiffeq.function_basis import ZonalSphericalHarmonics
from neurodiffeq.monitors import MonitorSphericalHarmonics
from networks import ModelFactory
from optimizers import OptimizerFactory
from monitors import MonitorCallbackFactory
from curriculum import CurriculumFactory
from conditions import ConditionFactory
from config import Config


class Session:
    def __init__(self, cfg):
        weighting_cfg = cfg.weighting
        if weighting_cfg is None:
            raise ValueError('weight_cfg = None')
        # set weighting
        self.weight_fns = {
            eq: ScalarComposition(
                fn=get_fn_by_name(cfg.type)(**cfg.__dict__['args']),
                alpha=cfg.weight,
            )
            for eq, cfg in weighting_cfg.items()
        }
        # set function basis
        self.harmonics_fn = ZonalSphericalHarmonics(degrees=list(cfg.pde.degrees.items()))
        # set equations
        self.pdes = ZonalHarmonicsNS(
            rho=cfg.pde.rho,
            mu=cfg.pde.mu,
            omega0=cfg.pde.omega0,
            omega1=cfg.pde.omega1,
            r0=cfg.pde.r0,
            r1=cfg.pde.r1,
            harmonics_fn=self.harmonics_fn,
        )
        # set networks
        self.nets = [ModelFactory.from_config(c) for _, c in cfg.network.items()]
        # set optimizer
        optimizer_getter = OptimizerFactory.from_config(cfg.optimizer)
        self.optimizer = optimizer_getter(chain.from_iterable(n.parameters() for n in self.nets))
        # set monitors
        self.monitor_callbacks = MonitorCallbackFactory.from_config(cfg.monitor, self.pdes, self.harmonics_fn)
        # set curriculum
        self.curriculum = CurriculumFactory.from_config(cfg)
        # set conditions
        self.conditions = ConditionFactory.from_config(cfg)
