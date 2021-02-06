from weighting import ScalarComposition, get_fn_by_name, WeightedResiduals
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
