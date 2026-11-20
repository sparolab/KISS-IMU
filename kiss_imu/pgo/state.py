"""Node state of the sliding window: (R, p, v, bg, ba) at time t.

Tangent ordering (15): [dtheta (right, body), dp (world), dv (world), dbg, dba].
"""

from dataclasses import dataclass

import numpy as np

from kiss_imu.geometry import so3

ND = 15
ROT, POS, VEL, BG, BA = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)


@dataclass
class NodeState:
    t: float
    R: np.ndarray
    p: np.ndarray
    v: np.ndarray
    bg: np.ndarray
    ba: np.ndarray

    def copy(self):
        return NodeState(self.t, self.R.copy(), self.p.copy(), self.v.copy(), self.bg.copy(), self.ba.copy())

    def retract(self, d):
        return NodeState(self.t, so3.project(self.R @ so3.exp(d[ROT])), self.p + d[POS], self.v + d[VEL],
                         self.bg + d[BG], self.ba + d[BA])

    def minus(self, other):
        """Tangent vector d with other.retract(d) ~ self."""
        d = np.empty(ND)
        d[ROT] = so3.log(other.R.T @ self.R)
        d[POS], d[VEL] = self.p - other.p, self.v - other.v
        d[BG], d[BA] = self.bg - other.bg, self.ba - other.ba
        return d

    @property
    def T(self):
        return so3.se3(self.R, self.p)
