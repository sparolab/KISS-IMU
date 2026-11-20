"""Factors of the sliding window. Each returns its whitened residual and, per node it
touches, the Jacobian with respect to that node's 15-d tangent (pgo/state.py).

Add a factor type by subclassing Factor and implementing linearize(states).
"""

import numpy as np

from kiss_imu.geometry import so3
from kiss_imu.imu.preintegration import GRAVITY

from .state import BA, BG, ND, POS, ROT, VEL


def sqrt_info(cov, floor=1e-15):
    """W with W^T W = cov^-1 (eigen-decomposition, tolerant of near-singular cov)."""
    s, V = np.linalg.eigh(0.5 * (cov + cov.T))
    return (V / np.sqrt(np.maximum(s, floor))).T


def sqrt_of_info(info, floor=0.0):
    """W with W^T W = info."""
    s, V = np.linalg.eigh(0.5 * (info + info.T))
    return (V * np.sqrt(np.maximum(s, floor))).T


def huber_weight(r, delta):
    n = np.linalg.norm(r)
    return 1.0 if n <= delta else np.sqrt(delta / n)


class Factor:
    keys = ()
    kind = 'factor'

    def linearize(self, states):
        """states: {key: NodeState}. Returns (r, {key: J (dim x 15)})."""
        raise NotImplementedError

    def cost(self, states):
        r, _ = self.linearize(states)
        return 0.5 * float(r @ r)


class PriorFactor(Factor):
    """Full-state prior on one node, diagonal sigmas [rot, pos, vel, bg, ba] (3 each, or scalars);
    only [rot, pos, vel] for graphs without bias states."""
    kind = 'prior'

    def __init__(self, key, state, sigmas):
        self.keys = (key,)
        self.x0 = state.copy()
        sig = np.concatenate([np.broadcast_to(np.asarray(s, float), 3) for s in sigmas])
        if len(sig) not in (9, ND):
            raise ValueError(f'prior sigmas: 3 or 5 groups, got {len(sigmas)}')
        self.n = len(sig)
        self.W = np.diag(1.0 / sig)

    def linearize(self, states):
        x = states[self.keys[0]]
        d = x.minus(self.x0)[:self.n]
        J = np.eye(ND)[:self.n]
        J[ROT, ROT] = so3.right_jacobian_inv(d[ROT])
        return self.W @ d, {self.keys[0]: self.W @ J}


class MarginalPrior(Factor):
    """Linear prior left by marginalisation: r = J (x [-] x_lin) + r0 over its nodes."""
    kind = 'marginal'

    def __init__(self, keys, lin_states, J, r0, nd=ND):
        self.keys = tuple(keys)
        self.lin = {k: lin_states[k].copy() for k in keys}
        self.J, self.r0, self.nd = J, r0, nd

    def linearize(self, states):
        nd = self.nd
        dx, jac = [], {}
        for n, k in enumerate(self.keys):
            d = states[k].minus(self.lin[k])[:nd]
            dx.append(d)
            D = np.eye(ND)[:nd]
            D[ROT, ROT] = so3.right_jacobian_inv(d[ROT])
            jac[k] = self.J[:, n * nd:(n + 1) * nd] @ D
        return self.J @ np.concatenate(dx) + self.r0, jac


class IMUFactor(Factor):
    """Preintegrated IMU between nodes i and j, first-order bias correction."""
    kind = 'imu'

    def __init__(self, i, j, pre, cov_scale=1.0):
        self.keys = (i, j)
        self.pre = pre
        self.W = sqrt_info(pre.cov * cov_scale)

    def linearize(self, states):
        xi, xj = states[self.keys[0]], states[self.keys[1]]
        pre, dt, g = self.pre, self.pre.dt, GRAVITY
        dbg = xi.bg - pre.bg
        dR, dV, dP = pre.corrected(xi.bg, xi.ba)
        RiT = xi.R.T
        e_R = so3.log(dR.T @ RiT @ xj.R)
        u_v = RiT @ (xj.v - xi.v - g * dt)
        u_p = RiT @ (xj.p - xi.p - xi.v * dt - 0.5 * g * dt * dt)
        r = np.concatenate([e_R, u_v - dV, u_p - dP])

        Jri = so3.right_jacobian_inv(e_R)
        Ji, Jj = np.zeros((9, ND)), np.zeros((9, ND))
        Ji[0:3, ROT] = -Jri @ xj.R.T @ xi.R
        Ji[0:3, BG] = -Jri @ so3.exp(e_R).T @ so3.right_jacobian(pre.dR_dbg @ dbg) @ pre.dR_dbg
        Jj[0:3, ROT] = Jri
        Ji[3:6, ROT] = so3.hat(u_v)
        Ji[3:6, VEL] = -RiT
        Ji[3:6, BG] = -pre.dV_dbg
        Ji[3:6, BA] = -pre.dV_dba
        Jj[3:6, VEL] = RiT
        Ji[6:9, ROT] = so3.hat(u_p)
        Ji[6:9, POS] = -RiT
        Ji[6:9, VEL] = -RiT * dt
        Ji[6:9, BG] = -pre.dP_dbg
        Ji[6:9, BA] = -pre.dP_dba
        Jj[6:9, POS] = RiT
        W = self.W
        return W @ r, {self.keys[0]: W @ Ji, self.keys[1]: W @ Jj}


class BiasFactor(Factor):
    """Random walk of the gyro / accelerometer biases between nodes i and j."""
    kind = 'bias'

    def __init__(self, i, j, dt, gyr_rw, acc_rw, jump=None):
        """jump: expected [dbg, dba] from i to j (zero for a random walk), e.g. at a switch of the IMU stream
        whose corrections already contain part of the bias."""
        self.keys = (i, j)
        sd = np.sqrt(max(dt, 1e-6))
        self.w = np.r_[np.full(3, 1.0 / (gyr_rw * sd)), np.full(3, 1.0 / (acc_rw * sd))]
        self.jump = np.zeros(6) if jump is None else np.asarray(jump, float)

    def linearize(self, states):
        xi, xj = states[self.keys[0]], states[self.keys[1]]
        r = np.r_[xj.bg - xi.bg, xj.ba - xi.ba] - self.jump
        Ji, Jj = np.zeros((6, ND)), np.zeros((6, ND))
        Ji[:, 9:15] = -np.eye(6)
        Jj[:, 9:15] = np.eye(6)
        W = self.w[:, None]
        return self.w * r, {self.keys[0]: W * Ji, self.keys[1]: W * Jj}


class PoseFactor(Factor):
    """World pose of one node from scan-to-map registration.

    r = [Log(R_m^T R), p - p_m], information `info` (6x6, same ordering), optional Huber."""
    kind = 'lo'

    def __init__(self, key, R_m, p_m, info, huber=None):
        self.keys = (key,)
        self.R_m, self.p_m = R_m, p_m
        self.W = sqrt_of_info(info)
        self.huber = huber

    def linearize(self, states):
        x = states[self.keys[0]]
        e_R = so3.log(self.R_m.T @ x.R)
        r = np.r_[e_R, x.p - self.p_m]
        J = np.zeros((6, ND))
        J[0:3, ROT] = so3.right_jacobian_inv(e_R)
        J[3:6, POS] = np.eye(3)
        r, J = self.W @ r, self.W @ J
        if self.huber:
            w = huber_weight(r, self.huber)
            r, J = w * r, w * J
        return r, {self.keys[0]: J}
