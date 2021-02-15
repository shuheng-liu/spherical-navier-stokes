from spherical_navier_stokes.config import Config
from neurodiffeq.callbacks import EveCallback, MonitorCallback, ReportOnFitCallback, CheckpointCallback
from neurodiffeq.callbacks import TrueCallback, FalseCallback, AndCallback, OrCallback, NotCallback, XorCallback
from neurodiffeq.callbacks import OnFirstLocal, OnLastLocal, OnFirstGlobal, PeriodGlobal, PeriodLocal
from neurodiffeq.callbacks import RepeatedMetricDiverge, RepeatedMetricConverge, RepeatedMetricUp, RepeatedMetricDown


class CallbackFactory:
    operator_callbacks = {
        'and': AndCallback,
        'or': OrCallback,
        'not': NotCallback,
        'xor': XorCallback,
    }

    condition_callbacks = {
        'on_first_local': OnFirstLocal,
        'on_first_global': OnFirstGlobal,
        'on_last_local': OnLastLocal,
        'period_local': PeriodLocal,
        'period_global': PeriodGlobal,
        'repeated_metric_up': RepeatedMetricUp,
        'repeated_metric_down': RepeatedMetricDown,
        'repeated_metric_converge': RepeatedMetricConverge,
        'repeated_metric_diverge': RepeatedMetricDiverge,
        'true': TrueCallback,
        'false': FalseCallback,
    }

    action_callbacks = {
        'eve': EveCallback,
        'monitor': MonitorCallback,
        'report': ReportOnFitCallback,
        'checkpoint': CheckpointCallback,
    }

    @staticmethod
    def from_config(cfg, logger=None):
        if not isinstance(cfg, Config):
            return cfg

        if cfg.type is None:
            return Config.to_builtin(cfg)

        if cfg.args:
            args = [CallbackFactory.from_config(arg, logger=logger) for arg in cfg.args.items()]
        else:
            args = []

        if cfg.kwargs:
            kwargs = {kw: CallbackFactory.from_config(arg, logger=logger) for kw, arg in cfg.kwargs.items()}
        else:
            kwargs = {}

        if cfg.type in CallbackFactory.operator_callbacks:
            if cfg.type == 'not':
                return NotCallback(args[0], logger=logger)
            CbClass = CallbackFactory.operator_callbacks[cfg.type]
            callback = CbClass(args, logger=logger, **kwargs)
        elif cfg.type in CallbackFactory.condition_callbacks:
            CbClass = CallbackFactory.condition_callbacks[cfg.type]
            callback = CbClass(*args, logger=logger, **kwargs)
        elif cfg.type in CallbackFactory.action_callbacks:
            CbClass = CallbackFactory.action_callbacks[cfg.type]
            callback = CbClass(*args, logger=logger, **kwargs)
        else:
            raise ValueError(f'Unknown callback type {cfg.type}')

        if cfg.action:
            action_callback = CallbackFactory.from_config(cfg.action, logger=logger)
            callback.set_action_callback(action_callback)
        return callback
