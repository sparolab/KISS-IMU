from dataclasses import dataclass, field

import numpy as np

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def build_registration(cfg):
    cfg = dict(cfg)
    name = cfg.pop('name')
    if name not in REGISTRY:
        raise KeyError(f'unknown registration {name!r}; known: {sorted(REGISTRY)}')
    return REGISTRY[name](**cfg)


@dataclass
class RegistrationResult:
    T: np.ndarray                     # target <- source, 4x4
    converged: bool
    H: np.ndarray = None              # 6x6 Hessian, [rot, trans], right perturbation T Exp(d); None if unavailable
    num_inliers: int = 0
    num_source: int = 0
    error: float = float('nan')
    iterations: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def inlier_ratio(self):
        return self.num_inliers / max(self.num_source, 1)


class Registration:
    """prepare_* turn an (N, 3) array into whatever the backend aligns; align returns
    a RegistrationResult for target <- source starting from T_init."""

    def prepare_source(self, points):
        raise NotImplementedError

    def prepare_target(self, points):
        raise NotImplementedError

    def align(self, source, target, T_init):
        raise NotImplementedError
