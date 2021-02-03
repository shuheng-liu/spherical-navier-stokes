from abc import ABC, abstractmethod
import torch


class WeightedResiduals:
    def __init__(self, eq_system, weight_fns):
        self.eq_system = eq_system
        self.weight_fns = weight_fns

    def __call__(self, *args, **kwargs):
        return [
            eq * w(*args, **kwargs)
            for eq, w in zip(self.eq_system(*args, **kwargs), self.weight_fns)
        ]


class WeightFunction(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Identity(WeightFunction):
    def __call__(self, *args, **kwargs):
        return 1.0


class SoftStep(WeightFunction):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, r, *args, **kwargs):
        return torch.sigmoid(self.a * r - self.b / r)


class ScalarComposition(WeightFunction):
    def __init__(self, fn, alpha):
        self.fn = fn
        self.alpha = alpha

    def __call__(self, *args, **kwargs):
        return self.alpha * self.fn(*args, **kwargs)
