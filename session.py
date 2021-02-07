from itertools import chain
from weighting import ScalarComposition, get_fn_by_name, WeightedResiduals
from equations import ZonalHarmonicsNS
from neurodiffeq.function_basis import ZonalSphericalHarmonics
from networks import ModelFactory
from optimizer import OptimizerFactory
from config import Config


class Session:
    def set_weighting(self, cfg):
        weighting_cfg = cfg.weighting
        if weighting_cfg is None:
            raise ValueError('weight_cfg = None')

        self.weight_fns = {
            eq: ScalarComposition(
                fn=get_fn_by_name(cfg.type)(**cfg.__dict__['args']),
                alpha=cfg.weight,
            )
            for eq, cfg in weighting_cfg.items()
        }

    def set_equations(self, cfg):
        pde_cfg = cfg.pde
        self.pdes = ZonalHarmonicsNS(
            rho=pde_cfg.rho,
            mu=pde_cfg.mu,
            omega0=pde_cfg.omega0,
            omega1=pde_cfg.omega1,
            r0=pde_cfg.r0,
            r1=pde_cfg.r1,
            harmonics_fn=ZonalSphericalHarmonics(degrees=list(pde_cfg.degrees.items())),
        )

    def set_networks(self, cfg):
        self.nets = [ModelFactory.from_config(c) for _, c in cfg.network.items()]

    def set_optimizer(self, cfg):
        optimizer_getter = OptimizerFactory.from_config(cfg.optimizer)
        self.optimizer = optimizer_getter(chain.from_iterable(n.parameters() for n in self.nets))
