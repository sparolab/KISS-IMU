# utils/gmm_module.py
import os
import numpy as np
import joblib
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import torch


class GmmModule:
    def __init__(self,
                 train_packs: List[Tuple[str, object, object]],
                 K: Optional[int] = None,
                 win_sec: float = 0.2,
                 random_state: int = 0,
                 n_init: int = 5):
        self.train_packs = train_packs
        self.K = K
        self.win_sec = win_sec
        self.random_state = random_state
        self.n_init = n_init
        self.scaler: Optional[StandardScaler] = None
        self.gmm: Optional[GaussianMixture] = None

    # ===================== Public APIs =====================
    def fit(self):
        X_list = []
        for seq, ds, _ in self.train_packs:
            Xi = self._features_from_dataset(ds)  # (Ni,2)
            if Xi is None or len(Xi) == 0:
                continue
            X_list.append(Xi)
        if len(X_list) == 0:
            raise ValueError("No features gathered from train_packs.")
        X_all = np.vstack(X_list)  # (Ntot,2)

        self.scaler = StandardScaler().fit(X_all)
        Xn = self.scaler.transform(X_all)

        if self.K is None:
            best_bic, best = None, None
            for k in range(2, 8):
                g = GaussianMixture(
                    n_components=k, covariance_type="full",
                    random_state=self.random_state, n_init=self.n_init
                )
                g.fit(Xn)
                bic = g.bic(Xn)
                if best_bic is None or bic < best_bic:
                    best_bic, best = bic, g
            self.gmm = best
        else:
            self.gmm = GaussianMixture(
                n_components=self.K, covariance_type="full",
                random_state=self.random_state, n_init=self.n_init
            ).fit(Xn)
        self.K = self.gmm.n_components
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N,2) = [[lin_speed, ang_speed], ...]"""
        self._ensure_fitted()
        Xn = self.scaler.transform(np.asarray(X))
        return self.gmm.predict(Xn)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        Xn = self.scaler.transform(np.asarray(X))
        return self.gmm.predict_proba(Xn)

    def predict_window(self,
                   imu_ts: np.ndarray,
                   accels: np.ndarray,
                   gyros: np.ndarray,
                   gravity: Optional[np.ndarray] = None,
                   reduce: str = "median",          # "last" | "mean" | "median" | "tail-mean" | "max" | "index" | "mode" | "soft-mode"
                   idx: Optional[int] = None,       # position used when reduce="index"
                   tail_ratio: float = 0.25,        # used by reduce="tail-mean" (tail-segment fraction)
                   return_proba: bool = False):

        lin_sp, ang_sp = self._imu_linear_angular_speed(imu_ts, accels, gyros, gravity)
        n = len(lin_sp)
        if n == 0:
            raise ValueError("Empty IMU window is given.")

        reduce = (reduce or "last").lower()

        # --- voting-based branch (uses the whole window) ---
        if reduce in ("mode", "soft-mode"):
            X_win = np.stack([lin_sp, ang_sp], axis=1)    # (n,2)

            if reduce == "mode":
                labels = self.predict(X_win)              # (n,)
                comp_id = int(np.bincount(labels).argmax())
                if return_proba:
                    P = self.predict_proba(X_win)         # (n,K)
                    proba = P.mean(axis=0)                
                    return comp_id, proba
                else:
                    return comp_id

            else:  # "soft-mode"
                P = self.predict_proba(X_win)             # (n,K)
                proba = P.mean(axis=0)                    
                comp_id = int(np.argmax(proba))
                if return_proba:
                    return comp_id, proba
                else:
                    return comp_id

        if reduce == "last":
            feat = np.array([[lin_sp[-1], ang_sp[-1]]], dtype=np.float64)
        elif reduce == "mean":
            feat = np.array([[lin_sp.mean(), ang_sp.mean()]], dtype=np.float64)
        elif reduce == "median":
            feat = np.array([[np.median(lin_sp), np.median(ang_sp)]], dtype=np.float64)
        elif reduce == "tail-mean":
            m = max(1, int(round(n * float(tail_ratio))))
            feat = np.array([[lin_sp[-m:].mean(), ang_sp[-m:].mean()]], dtype=np.float64)
        elif reduce == "max":
            feat = np.array([[lin_sp.max(), ang_sp.max()]], dtype=np.float64)
        elif reduce == "index":
            if idx is None:
                raise ValueError("reduce='index' requires idx to be provided.")
            i = int(np.clip(idx, 0, n - 1))
            feat = np.array([[lin_sp[i], ang_sp[i]]], dtype=np.float64)
        else:
            raise ValueError(f"Unknown reduce='{reduce}'")

        if return_proba:
            proba = self.predict_proba(feat)[0]           # (K,)
            comp_id = int(np.argmax(proba))
            return comp_id, proba
        else:
            return int(self.predict(feat)[0])

    def save(self, path: str):
        self._ensure_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "gmm": self.gmm}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.scaler = obj["scaler"]
        self.gmm = obj["gmm"]
        return self

    # ===================== Internal: feature building =====================
    def _features_from_dataset(self, ds) -> np.ndarray:
        t = np.asarray(ds.imu_ts, dtype=np.float64)
        acc = np.asarray(ds.accels, dtype=np.float64)
        gyr = np.asarray(ds.gyros, dtype=np.float64)
        gvec = np.asarray(getattr(ds, "gravity", np.array([0.,0.,9.81], dtype=np.float32)), dtype=np.float64)

        lin_sp, ang_sp = self._imu_linear_angular_speed(t, acc, gyr, gvec)
        return np.stack([lin_sp, ang_sp], axis=1)  # (N,2)

    # ===================== Internal: kinematics over IMU =====================
    def compute_component_weights(self,
                                  source: str = "train",     # "train" | "mixing"
                                  method: str = "effective", # "inverse" | "sqrt_inv" | "effective"
                                  beta: float = 0.999,       # used by method="effective" (Class-Balanced Loss)
                                  normalize: str = "mean1",  # "mean1" | "sum1" | "none"
                                  clamp: Tuple[float, float] = (0.1, 10.0),
                                  recalc: bool = True) -> np.ndarray:

        self._ensure_fitted()
        K = self.gmm.weights_.shape[0]

        if source == "mixing":
            Nk = np.maximum(self.gmm.weights_ * 1.0, 1e-12)
            Nk = Nk * 1e6 
        elif source == "train":
            if recalc or not hasattr(self, "_Nk_soft_"):
                X_list = []
                for _, ds, _ in self.train_packs:
                    Xi = self._features_from_dataset(ds)
                    if Xi is None or len(Xi) == 0:
                        continue
                    X_list.append(Xi)
                if len(X_list) == 0:
                    raise ValueError("No features found to compute frequencies.")
                X_all = np.vstack(X_list)
                Xn = self.scaler.transform(X_all)
                resp = self.gmm.predict_proba(Xn)   # (N,K)
                Nk = resp.sum(axis=0)               # soft count
                self._Nk_soft_ = Nk
            else:
                Nk = self._Nk_soft_
        else:
            raise ValueError(f"Unknown source: {source}")

        Nk = np.asarray(Nk, dtype=np.float64)
        eps = 1e-12
        if method == "inverse":
            w = 1.0 / np.clip(Nk, eps, None)
        elif method == "sqrt_inv":
            w = 1.0 / np.sqrt(np.clip(Nk, eps, None))
        elif method == "effective":
            w = (1.0 - beta) / (1.0 - np.power(beta, np.clip(Nk, 1.0, None)))
        else:
            raise ValueError(f"Unknown method: {method}")

        if normalize == "mean1":
            w = w / (w.mean() + eps)
        elif normalize == "sum1":
            w = w / (w.sum() + eps)
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize: {normalize}")

        w = np.clip(w, clamp[0], clamp[1])

        self._comp_weights_ = w.copy()
        return w

    def sample_weights_from_labels(self,
                                   comp_ids: np.ndarray,
                                   comp_weights: Optional[np.ndarray] = None,
                                   normalize: str = "mean1") -> np.ndarray:

        self._ensure_fitted()
        comp_ids = np.asarray(comp_ids).astype(int)
        if comp_weights is None:
            if hasattr(self, "_comp_weights_"):
                comp_weights = self._comp_weights_
            else:
                comp_weights = self.compute_component_weights(source="train", method="effective")
        w = comp_weights[comp_ids]

        if normalize == "mean1":
            w = w / (w.mean() + 1e-12)
        elif normalize == "sum1":
            w = w / (w.sum() + 1e-12)
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize: {normalize}")
        return w

    def sample_weights_from_proba(self,
                                  comp_proba: np.ndarray,
                                  comp_weights: Optional[np.ndarray] = None,
                                  normalize: str = "mean1") -> np.ndarray:

        self._ensure_fitted()
        P = np.asarray(comp_proba, dtype=np.float64)
        if comp_weights is None:
            if hasattr(self, "_comp_weights_"):
                comp_weights = self._comp_weights_
            else:
                comp_weights = self.compute_component_weights(source="train", method="effective")
        w = (P @ comp_weights.reshape(-1, 1)).reshape(-1)

        if normalize == "mean1":
            w = w / (w.mean() + 1e-12)
        elif normalize == "sum1":
            w = w / (w.sum() + 1e-12)
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize: {normalize}")
        return w

    def torch_weights_from_labels(self,
                                  comp_ids: np.ndarray,
                                  device=None,
                                  dtype=None) -> "torch.Tensor":

        if torch is None:
            raise RuntimeError("PyTorch not available.")
        w = self.sample_weights_from_labels(comp_ids)
        t = torch.from_numpy(w)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t
    
    @staticmethod
    def _yaw_only_Rz(yaw: np.ndarray) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        Rw = np.stack([
            np.stack([c, -s, np.zeros_like(yaw)], axis=-1),
            np.stack([s,  c, np.zeros_like(yaw)], axis=-1),
            np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)], axis=-1),
        ], axis=-2)  # (...,3,3)
        return Rw

    @staticmethod
    def _moving_sum(x: np.ndarray, L: int) -> np.ndarray:
        k = np.ones(L, dtype=np.float64)
        if x.ndim == 1:
            return np.convolve(x, k, mode='same')
        else:
            return np.stack([np.convolve(x[:, i], k, mode='same') for i in range(x.shape[1])], axis=1)

    def _imu_linear_angular_speed(self,
                                  t_imu: np.ndarray,
                                  acc_b: np.ndarray,
                                  gyr_b: np.ndarray,
                                  g_vec: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:

        if hasattr(t_imu, 'cpu'):
            t_imu = t_imu.cpu().numpy()
        if hasattr(acc_b, 'cpu'):
            acc_b = acc_b.cpu().numpy()
        if hasattr(gyr_b, 'cpu'):
            gyr_b = gyr_b.cpu().numpy()
        if g_vec is not None and hasattr(g_vec, 'cpu'):
            g_vec = g_vec.cpu().numpy()
        
        t = np.asarray(t_imu, dtype=np.float64)
        acc_b = np.asarray(acc_b, dtype=np.float64)
        gyr_b = np.asarray(gyr_b, dtype=np.float64)
        if g_vec is None:
            g_vec = np.array([0., 0., 9.81], dtype=np.float64)
        else:
            g_vec = np.asarray(g_vec, dtype=np.float64)

        dt = np.diff(t, prepend=t[0])
        if not np.any(dt > 0):
            dt = np.ones_like(dt) * 1e-3
        else:
            pos_dt = dt[dt > 0]
            median_dt = np.median(pos_dt) if len(pos_dt) else 1e-3
            dt[dt <= 0] = median_dt

        yaw = np.cumsum(gyr_b[:, 2] * dt)
        R_wb = self._yaw_only_Rz(yaw)                 
        acc_w = (R_wb @ acc_b[..., None]).squeeze(-1)  
        acc_w = acc_w - g_vec                          

        dv  = acc_w * dt[:, None]
        dth = gyr_b * dt[:, None]

        L = max(1, int(round(self.win_sec / float(np.median(dt)))))
        sum_dt  = self._moving_sum(dt, L)
        sum_dv  = self._moving_sum(dv, L)
        sum_dth = self._moving_sum(dth, L)

        lin_speed = np.linalg.norm(sum_dv, axis=1)  / np.clip(sum_dt, 1e-6, None)
        ang_speed = np.linalg.norm(sum_dth, axis=1) / np.clip(sum_dt, 1e-6, None)
        return lin_speed, ang_speed

    # ===================== misc =====================
    def _ensure_fitted(self):
        if self.scaler is None or self.gmm is None:
            raise RuntimeError("GmmModule is not fitted. Call fit() or load().")
