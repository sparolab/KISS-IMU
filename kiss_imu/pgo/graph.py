"""Sliding-window factor graph: Levenberg-Marquardt over the window, Schur-complement
marginalisation of the oldest node into a MarginalPrior on its neighbours.

nd: estimated dimensions per node, the leading part of the 15-d tangent (pgo/state.py): 15 = R, p, v, bg, ba;
9 = R, p, v only (no bias states: the bias Jacobian columns are dropped, the biases stay where they started)."""

from collections import OrderedDict

import numpy as np

from .factors import MarginalPrior
from .state import ND


class SlidingWindowGraph:
    def __init__(self, size=10, iters=6, lm_lambda=1e-4, tol=1e-6, nd=ND):
        if nd not in (9, ND):
            raise ValueError(f'nd must be 9 (no biases) or {ND}, got {nd}')
        self.size, self.iters, self.lm_lambda0, self.tol, self.nd = size, iters, lm_lambda, tol, nd
        self.states = OrderedDict()            # key -> NodeState (oldest first)
        self.factors = []

    # ------------------------------------------------------------------ building
    def add_node(self, key, state):
        self.states[key] = state

    def add_factor(self, f):
        self.factors.append(f)

    def __len__(self):
        return len(self.states)

    @property
    def newest_key(self):
        return next(reversed(self.states))

    @property
    def oldest_key(self):
        return next(iter(self.states))

    # ------------------------------------------------------------------ solve
    def _system(self, factors, keys, states):
        nd = self.nd
        idx = {k: n for n, k in enumerate(keys)}
        H = np.zeros((nd * len(keys),) * 2)
        g = np.zeros(nd * len(keys))
        cost = 0.0
        for f in factors:
            r, jac = f.linearize(states)
            cost += 0.5 * float(r @ r)
            jac = {k: J[:, :nd] for k, J in jac.items()}
            for a, Ja in jac.items():
                sa = slice(nd * idx[a], nd * idx[a] + nd)
                g[sa] += Ja.T @ r
                for b, Jb in jac.items():
                    sb = slice(nd * idx[b], nd * idx[b] + nd)
                    H[sa, sb] += Ja.T @ Jb
        return H, g, cost

    def cost(self, states=None):
        states = states or self.states
        return sum(f.cost(states) for f in self.factors)

    def optimize(self):
        keys = list(self.states)
        lam = self.lm_lambda0
        H, g, cost = self._system(self.factors, keys, self.states)
        for _ in range(self.iters):
            A = H + lam * np.diag(np.diag(H) + 1e-9)
            try:
                d = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                d = np.linalg.lstsq(A, -g, rcond=None)[0]
            nd = self.nd
            trial = OrderedDict((k, self.states[k].retract(np.r_[d[nd * n:nd * n + nd], np.zeros(ND - nd)]))
                                for n, k in enumerate(keys))
            H_t, g_t, cost_t = self._system(self.factors, keys, trial)
            if cost_t < cost:
                self.states, H, g, cost = trial, H_t, g_t, cost_t
                lam = max(lam / 10, 1e-9)
                if np.linalg.norm(d) < self.tol:
                    break
            else:
                lam *= 10
                if lam > 1e8:
                    break
        self.H, self.H_keys = H, keys            # Gauss-Newton information at the solution (for covariances)
        return cost

    def marginal_cov(self, key):
        """(nd, nd) marginal covariance of one node from the last optimisation (inverse of the full H)."""
        nd = self.nd
        n = self.H_keys.index(key)
        Hr = self.H + 1e-9 * np.eye(len(self.H))
        C = np.linalg.inv(Hr)
        return C[nd * n:nd * n + nd, nd * n:nd * n + nd]

    # ------------------------------------------------------------------ marginalisation
    def marginalize_oldest(self, eig_floor=1e-12):
        """Remove the oldest node; returns its final state."""
        k0 = self.oldest_key
        touching = [f for f in self.factors if k0 in f.keys]
        others = [k for k in self.states if k != k0 and any(k in f.keys for f in touching)]
        state0 = self.states[k0]
        self.factors = [f for f in self.factors if k0 not in f.keys]
        if others:
            keys = [k0] + others
            H, g, _ = self._system(touching, keys, self.states)
            nd = self.nd
            Haa, Hab, Hbb = H[:nd, :nd], H[:nd, nd:], H[nd:, nd:]
            ga, gb = g[:nd], g[nd:]
            Haa_inv = np.linalg.pinv(Haa)
            Hm = Hbb - Hab.T @ Haa_inv @ Hab
            gm = gb - Hab.T @ Haa_inv @ ga
            s, V = np.linalg.eigh(0.5 * (Hm + Hm.T))
            keep = s > eig_floor * max(s.max(), 1.0)
            J = (V[:, keep] * np.sqrt(s[keep])).T
            r0 = (V[:, keep].T @ gm) / np.sqrt(s[keep])
            self.factors.append(MarginalPrior(others, {k: self.states[k] for k in others}, J, r0, nd=self.nd))
        del self.states[k0]
        return state0
