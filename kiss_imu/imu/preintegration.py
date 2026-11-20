"""On-manifold IMU preintegration (Forster et al., T-RO 2017) with bias Jacobians, numpy.

Gravity is world -z and the accelerometer reads specific force, so between nodes i, j

    R_j = R_i dR
    v_j = v_i + g dt + R_i dV
    p_j = p_i + v_i dt + 1/2 g dt^2 + R_i dP

Covariance is ordered [dtheta, dv, dp].
"""

import numpy as np

from kiss_imu.geometry import so3

GRAVITY = np.array([0.0, 0.0, -9.81])


class Preintegrated:
    """dR, dV, dP over [t0, t1] at the linearisation bias (bg, ba)."""

    def __init__(self, bg, ba):
        self.bg = np.asarray(bg, float).copy()
        self.ba = np.asarray(ba, float).copy()
        self.dR, self.dV, self.dP = np.eye(3), np.zeros(3), np.zeros(3)
        self.dt = 0.0
        self.cov = np.zeros((9, 9))
        self.dR_dbg = np.zeros((3, 3))
        self.dV_dbg, self.dV_dba = np.zeros((3, 3)), np.zeros((3, 3))
        self.dP_dbg, self.dP_dba = np.zeros((3, 3)), np.zeros((3, 3))

    def integrate(self, gyr, acc, dt, gyr_noise, acc_noise, gyr_var=None, acc_var=None):
        """One zero-order-hold step; *_noise are continuous densities (unit / sqrt(Hz)); *_var, when given,
        are this sample's per-axis variances ((unit)^2, i.e. density^2 / dt) and take their place."""
        w = gyr - self.bg
        a = acc - self.ba
        dR, Ra = self.dR, self.dR @ so3.hat(a)
        Exp = so3.exp(w * dt)
        Jr = so3.right_jacobian(w * dt)

        A = np.eye(9)
        A[0:3, 0:3] = Exp.T
        A[3:6, 0:3] = -Ra * dt
        A[6:9, 0:3] = -0.5 * Ra * dt * dt
        A[6:9, 3:6] = np.eye(3) * dt
        Bg = np.zeros((9, 3)); Bg[0:3] = Jr * dt
        Ba = np.zeros((9, 3)); Ba[3:6] = dR * dt; Ba[6:9] = 0.5 * dR * dt * dt
        gv = np.full(3, gyr_noise ** 2 / dt) if gyr_var is None else gyr_var
        av = np.full(3, acc_noise ** 2 / dt) if acc_var is None else acc_var
        self.cov = A @ self.cov @ A.T + (Bg * gv) @ Bg.T + (Ba * av) @ Ba.T

        # bias Jacobians use the old dR and dR_dbg, so they go first
        self.dP_dba += self.dV_dba * dt - 0.5 * dR * dt * dt
        self.dP_dbg += self.dV_dbg * dt - 0.5 * Ra @ self.dR_dbg * dt * dt
        self.dV_dba += -dR * dt
        self.dV_dbg += -Ra @ self.dR_dbg * dt
        self.dR_dbg = Exp.T @ self.dR_dbg - Jr * dt

        self.dP += self.dV * dt + 0.5 * dR @ a * dt * dt
        self.dV += dR @ a * dt
        self.dR = dR @ Exp
        self.dt += dt

    def corrected(self, bg, ba):
        """(dR, dV, dP) at a new bias, first order in the bias change."""
        dbg, dba = bg - self.bg, ba - self.ba
        dR = self.dR @ so3.exp(self.dR_dbg @ dbg)
        dV = self.dV + self.dV_dbg @ dbg + self.dV_dba @ dba
        dP = self.dP + self.dP_dbg @ dbg + self.dP_dba @ dba
        return dR, dV, dP


def preintegrate(ts, gyr, acc, t0, t1, bg, ba, gyr_noise, acc_noise, gyr_var=None, acc_var=None):
    """Integrate the samples over [t0, t1], each held until the next one (the sample
    in force at t0 is the last one at or before t0). gyr_var / acc_var: optional per-sample,
    per-axis variances aligned with ts (see kiss_imu.imu.noise) in place of the densities."""
    k0 = max(int(np.searchsorted(ts, t0, side='right')) - 1, 0)
    k1 = int(np.searchsorted(ts, t1, side='left'))
    knots = np.concatenate([[t0], ts[k0 + 1:k1], [t1]])
    pre = Preintegrated(bg, ba)
    for n in range(len(knots) - 1):
        dt = knots[n + 1] - knots[n]
        if dt > 0:
            k = k0 + n
            pre.integrate(gyr[k], acc[k], dt, gyr_noise, acc_noise,
                          None if gyr_var is None else gyr_var[k], None if acc_var is None else acc_var[k])
    return pre


def predict(Ri, pi, vi, pre, bg=None, ba=None):
    """Propagate a state through a preintegrated interval (at its own bias unless given)."""
    dR, dV, dP = (pre.dR, pre.dV, pre.dP) if bg is None else pre.corrected(bg, ba)
    dt = pre.dt
    Rj = Ri @ dR
    vj = vi + GRAVITY * dt + Ri @ dV
    pj = pi + vi * dt + 0.5 * GRAVITY * dt * dt + Ri @ dP
    return Rj, pj, vj
