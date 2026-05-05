# utils/convergence.py
import numpy as np

class CompConvergenceManager:
    def __init__(self, K, ema_beta=0.9, improve_tol=1e-3, abs_tol=None, patience=5, min_seen=100):
        self.K = int(K)
        self.beta = float(ema_beta)
        self.improve_tol = float(improve_tol)
        self.abs_tol = None if abs_tol is None else float(abs_tol)
        self.patience = int(patience)
        self.min_seen = int(min_seen)

        self.ema = np.full(self.K, np.inf, dtype=np.float64)
        self.best = np.full(self.K, np.inf, dtype=np.float64)
        self.count = np.zeros(self.K, dtype=np.int64)
        self.bad = np.zeros(self.K, dtype=np.int64)
        self._converged = np.zeros(self.K, dtype=bool)

    def is_converged(self, k):
        return bool(self._converged[int(k)])

    def all_converged(self, comp_ids):
        comp_ids = np.asarray(comp_ids, dtype=int)
        return np.all(self._converged[comp_ids])

    def update(self, comp_ids, per_sample_loss):
        comp_ids = np.asarray(comp_ids, dtype=int)
        losses = np.asarray(per_sample_loss, dtype=np.float64)
        
        # Ensure comp_ids and losses have the same length
        if len(comp_ids) != len(losses):
            print(f"Warning: comp_ids length ({len(comp_ids)}) != losses length ({len(losses)})")
            # Truncate to the shorter length
            min_len = min(len(comp_ids), len(losses))
            comp_ids = comp_ids[:min_len]
            losses = losses[:min_len]
        
        new_conv = []
        for k in np.unique(comp_ids):
            # Skip invalid component IDs
            if k >= self.K:
                print(f"Warning: Skipping invalid component ID {k} (>= {self.K})")
                continue
                
            idx = np.where(comp_ids == k)[0]
            if idx.size == 0:
                continue
            cur = float(np.mean(losses[idx]))
            if np.isinf(self.ema[k]):
                self.ema[k] = cur
            else:
                self.ema[k] = self.beta * self.ema[k] + (1 - self.beta) * cur

            self.count[k] += idx.size

            improved = self.best[k] - self.ema[k]
            if self.ema[k] < self.best[k]:
                self.best[k] = self.ema[k]
                self.bad[k] = 0
            else:
                self.bad[k] += 1

            cond_patience = (self.bad[k] >= self.patience) and (abs(improved) < self.improve_tol)
            cond_abs = (self.abs_tol is not None) and (self.ema[k] <= self.abs_tol)
            if (self.count[k] >= self.min_seen) and (cond_patience or cond_abs) and (not self._converged[k]):
                self._converged[k] = True
                print(f"\033[92m[INFO] component {k} optimized complete! ema={self.ema[k]:.6f}\033[0m")
                new_conv.append(int(k))
        return new_conv

    def reset(self, k=None):
        if k is None:
            self.__init__(self.K, self.beta, self.improve_tol, self.abs_tol, self.patience, self.min_seen)
        else:
            k = int(k)
            self.ema[k] = np.inf
            self.best[k] = np.inf
            self.count[k] = 0
            self.bad[k] = 0
            self._converged[k] = False
