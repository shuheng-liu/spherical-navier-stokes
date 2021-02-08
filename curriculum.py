import math
import torch
import numpy as np
from abc import ABC, abstractmethod
from neurodiffeq.solvers import BaseSolver
from neurodiffeq.generators import SamplerGenerator, GeneratorSpherical, Generator3D


class BaseCurriculumLearner(ABC):
    def __init__(self, n_curricula, train_gen_getter, valid_gen_getter, base_size, min_size=None, max_size=None):
        self.n_curricula = n_curricula
        self.gen_getter = dict(train=train_gen_getter, valid=valid_gen_getter)
        if isinstance(base_size, (tuple, list)):
            self.base_size = dict(train=base_size[0], valid=base_size[1])
        else:
            self.base_size = dict(train=base_size, valid=base_size)
        self.min_size = min_size
        self.max_size = max_size

    def clip_size(self, size):
        size = int(size)
        if self.min_size is not None and size < self.min_size:
            return self.min_size
        if self.max_size is not None and size > self.max_size:
            return self.max_size
        return size

    @abstractmethod
    def get_generator(self, curr_i, phase):
        pass

    @abstractmethod
    def get_size_or_shape(self, curr_i, phase):
        return self.clip_size(self.size[phase])

    def fit(self, solver: BaseSolver, epochs_per_curriculum, callbacks):
        for i in range(self.n_curricula):
            solver.generator['train'] = self.get_generator(i, phase='train')
            solver.generator['valid'] = self.get_generator(i, phase='valid')
            solver.fit(max_epochs=epochs_per_curriculum, callbacks=callbacks)


class RadialCurriculumLearner(BaseCurriculumLearner):
    def __init__(self, n_curricula, train_gen_getter, valid_gen_getter, base_size,
                 r_in_from, r_in_to, r_out_from, r_out_to,
                 advance_method='log', size_method='log', min_size=1, max_size=None):
        super(RadialCurriculumLearner, self).__init__(
            n_curricula=n_curricula,
            train_gen_getter=train_gen_getter,
            valid_gen_getter=valid_gen_getter,
            base_size=base_size,
            min_size=min_size,
            max_size=max_size,
        )
        if advance_method == "log":
            self.r_inners = np.logspace(np.log(r_in_from), np.log(r_in_to), num=n_curricula, base=math.e)
            self.r_outers = np.logspace(np.log(r_out_from), np.log(r_out_to), num=n_curricula, base=math.e)
        elif advance_method == 'linear':
            self.r_inners = np.linspace(r_in_from, r_in_to, num=n_curricula)
            self.r_outers = np.linspace(r_out_from, r_out_to, num=n_curricula)
        else:
            raise ValueError(f"Unknown `advance_method` {advance_method}")

        self.size_method = size_method

    def get_size_or_shape(self, curr_i, phase):
        if self.size_method == 'constant':
            return self.clip_size(self.base_size[phase])

        if isinstance(self.size_method, int):
            cdf = lambda r: r ** self.size_method
        elif self.size_method == 'log':
            cdf = np.log
        elif self.size_method == 'exponential':
            cdf = np.exp
        else:
            raise ValueError(f"Unknown `size_method` {self.size_method}")

        drs = cdf(self.r_outers) - cdf(self.r_inners)
        return self.clip_size(self.base_size[phase] * drs[curr_i] / drs.max())

    def get_generator(self, curr_i, phase):
        g = self.gen_getter[phase](
            self.get_size_or_shape(curr_i, phase),
            self.r_inners[curr_i],
            self.r_outers[curr_i],
        )
        return g if isinstance(g, SamplerGenerator) else SamplerGenerator(g)


class CurriculumFactory:
    @staticmethod
    def from_config(root_cfg):
        def generator_meta_getter(phase):
            generator_config = getattr(root_cfg.curriculum, phase).generator

            def generator_getter(size, r0, r1):
                if generator_config.type.lower() == 'spherical':
                    return GeneratorSpherical(size, r0, r1, method=generator_config.method)
                else:
                    raise ValueError("")

            return generator_getter

        return RadialCurriculumLearner(
            n_curricula=root_cfg.curriculum.n_curricula,
            train_gen_getter=generator_meta_getter('train'),
            valid_gen_getter=generator_meta_getter('valid'),
            base_size=root_cfg.curriculum.base_size,
            r_in_from=root_cfg.curriculum.r0_start,
            r_in_to=root_cfg.pde.r0,
            r_out_from=root_cfg.curriculum.r1_start,
            r_out_to=root_cfg.pde.r1,
            advance_method=root_cfg.curriculum.advance_method,
            size_method=root_cfg.curriculum.size_method,
            min_size=root_cfg.curriculum.min_size,
            max_size=root_cfg.curriculum.max_size,
        )
