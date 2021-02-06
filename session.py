from weighting import ScalarComposition, get_fn_by_name, WeightedResiduals
from equations import ZonalHarmonicsNS
from neurodiffeq.function_basis import ZonalSphericalHarmonics
from config import Config


class Session:
    def set_weighting(self, weighting_cfg):
        if weighting_cfg is None:
            raise ValueError('weight_cfg = None')

        self.weight_fns = {
            eq: ScalarComposition(
                fn=get_fn_by_name(cfg.type)(**cfg.__dict__['args']),
                alpha=cfg.weight,
            )
            for eq, cfg in weighting_cfg.items()
        }

    def set_equations(self, pde_cfg, numerical_cfg):
        harmonics_fn = ZonalSphericalHarmonics(degrees=numerical_cfg.degrees)
        self.pdes = ZonalHarmonicsNS(
            rho=pde_cfg.rho,
            mu=pde_cfg.mu,
            omega0=pde_cfg.omega0,
            omega1=pde_cfg.omega1,
            r0=pde_cfg.r0,
            r1=pde_cfg.r1,
            harmonics_fn=harmonics_fn,
        )
