import os
import sys
import logging
from itertools import chain
from spherical_navier_stokes.weighting import ScalarComposition, get_fn_by_name
from spherical_navier_stokes.equations import ZonalHarmonicsNS
from neurodiffeq.function_basis import ZonalSphericalHarmonics
from neurodiffeq.solvers import SolverSpherical
from spherical_navier_stokes.networks import ModelFactory
from spherical_navier_stokes.optimizers import OptimizerFactory
from spherical_navier_stokes.monitors import MonitorCallbackFactory
from spherical_navier_stokes.curriculum import CurriculumFactory
from spherical_navier_stokes.conditions import ConditionFactory
from spherical_navier_stokes.utils import dump, timestr
from neurodiffeq.callbacks import ReportOnFitCallback, CheckpointCallback
from pathlib import Path


class Session:
    def __init__(self, cfg):
        self.root_cfg = cfg

        self.logger.propagate = 0
        log_path = Path(cfg.meta.base_path) / cfg.meta.log_path / (cfg._name + '.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(filename=log_path)
        handler.setFormatter(logging.Formatter(cfg.meta.log_format))
        handler.setLevel(cfg.meta.log_level)
        self.logger.addHandler(handler)

        if cfg.meta.log_console:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(logging.Formatter(cfg.meta.log_console_format))
            handler.setLevel(cfg.meta.log_console_level)
            self.logger.addHandler(handler)

        self.logger.info('\n' + cfg.to_yml())

        if cfg.callback is None:
            self.callbacks = [
                ReportOnFitCallback(logger=self.logger),
                CheckpointCallback(ckpt_dir=cfg.meta.output_path, logger=self.logger),
            ]

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
        network_order = ['ur', 'utheta', 'uphi', 'p']
        nets = {k: ModelFactory.from_config(c) for k, c in cfg.network.items()}
        self.nets = [nets.get(k) for k in network_order]
        # set optimizer
        optimizer_getter = OptimizerFactory.from_config(cfg.optimizer)
        self.optimizer = optimizer_getter(chain.from_iterable(n.parameters() for n in self.nets))
        # set monitors
        self.monitor_callbacks = MonitorCallbackFactory.from_config(cfg, self.pdes, self.harmonics_fn)
        # set curriculum
        self.curriculum = CurriculumFactory.from_config(cfg)
        # set conditions
        conditions = ConditionFactory.from_config(cfg)
        self.conditions = [conditions.get(k) for k in network_order]
        # set solver
        self.solver = SolverSpherical(
            pde_system=self.pdes,
            conditions=self.conditions,
            r_min=cfg.pde.r0,
            r_max=cfg.pde.r1,
            nets=self.nets,
            optimizer=self.optimizer,
            n_batches_train=cfg.curriculum.train.n_batches,
            n_batches_valid=cfg.curriculum.valid.n_batches,
        )

    def fit(self):
        self.curriculum.fit(
            solver=self.solver,
            epochs_per_curriculum=self.root_cfg.curriculum.epochs_per_curriculum,
            callbacks=self.monitor_callbacks + self.callbacks,
        )

    def dump(self, path=None):
        path = Path(path or os.path.join(self.root_cfg.meta.base_path, self.root_cfg.meta.output_path))
        dump(self.solver.get_internals(), path / (timestr() + ".internals"))
        self.root_cfg.to_yml_file(path / 'config.yaml')

    @property
    def logger(self):
        return logging.getLogger(self.root_cfg._name)
