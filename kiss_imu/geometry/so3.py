"""SO(3) helpers in numpy (single 3-vectors / 3x3 matrices), used by the PGO.

Right-perturbation convention throughout: R <- R Exp(d).
"""

import numpy as np

_EPS = 1e-8


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-6:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def log(Rm):
    c = np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if th < 1e-6:
        return 0.5 * v
    if np.pi - th < 1e-6:                       # near pi: axis from the symmetric part
        B = 0.5 * (Rm + np.eye(3))
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / np.sqrt(max(B[k, k], _EPS))
        return th * axis
    return th / (2 * np.sin(th)) * v


def right_jacobian(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-5:
        return np.eye(3) - 0.5 * K + K @ K / 6
    return np.eye(3) - (1 - np.cos(th)) / th ** 2 * K + (th - np.sin(th)) / th ** 3 * K @ K


def right_jacobian_inv(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-5:
        return np.eye(3) + 0.5 * K + K @ K / 12
    return np.eye(3) + 0.5 * K + (1 / th ** 2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * K @ K


def project(Rm):
    """Nearest rotation matrix."""
    U, _, Vt = np.linalg.svd(Rm)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def se3(Rm, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, p
    return T
