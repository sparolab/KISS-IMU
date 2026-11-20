"""Per scan pair, pick the best of several pseudo-label trajectories by point-cloud overlap (no GT).

KISS-IMU (the ICRA 2026 code, branch icra2026: src/train.py) labels every scan pair with the ICP motion or the motion of a
PGO that uses the network's corrected IMU, whichever aligns the two scans better (symmetric overlap,
utils/overlap_score.py). As the network gets better on a motion, the PGO labels of that motion get better and
win the vote, so training on rare motion improves exactly the labels of rare motion. Here the candidates are
whole label trajectories at the scan stamps (the LO poses of the raw-IMU run, and the LO / smoothed poses of
a re-run with the corrected IMU); the chosen relative motions are chained into a new label trajectory.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def voxel_down(x, voxel):
    if voxel is None or voxel <= 0 or len(x) == 0:
        return x
    k = np.floor(x / voxel).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return x[np.sort(idx)]


def symmetric_overlap(a, b, thr):
    """Fraction of the points of a and b with a neighbour within thr in the other cloud."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    da, _ = cKDTree(b).query(a, k=1, distance_upper_bound=thr)
    db, _ = cKDTree(a).query(b, k=1, distance_upper_bound=thr)
    return float((np.sum(da < thr) + np.sum(db < thr)) / (len(a) + len(b)))


def _T(P):
    """TUM rows (n, 8) -> rotations, positions."""
    return Rotation.from_quat(P[:, 4:8]), P[:, 1:4]


def select_labels(seq, cands, invalid, thr=0.3, voxel=0.25, margin=0.0, max_points=20000, seed=0):
    """seq: the sequence (scans in the IMU frame, seq.scan_t). cands: list of (n, 8) TUM label trajectories;
    every one is interpolated to the stamps of cands[0]. invalid: list of invalid-stamp arrays (one per
    candidate). Returns (P (n, 8), invalid_t, info) with P chained from cands[0][0] through the chosen pair
    motions; a pair whose chosen candidate has an invalid end keeps that stamp invalid. margin: another
    candidate replaces cands[0] only if its overlap is higher by more than margin (near-ties keep the current
    label instead of mixing trajectories on overlap noise)."""
    rng = np.random.default_rng(seed)
    t = cands[0][:, 0]
    rel, okc = [], []
    for P, inv in zip(cands, invalid):
        if len(P) != len(t) or np.abs(P[:, 0] - t).max() > 1e-4:          # align to the reference stamps
            j = np.searchsorted(P[:, 0], t).clip(1, len(P) - 1)
            j = np.where(np.abs(P[j - 1, 0] - t) < np.abs(P[j, 0] - t), j - 1, j)
            have = np.abs(P[j, 0] - t) < 1e-3
            P = P[j]
        else:
            have = np.ones(len(t), bool)
        Rr, p = _T(P)
        dR = Rr[:-1].inv() * Rr[1:]
        dp = Rr[:-1].inv().apply(p[1:] - p[:-1])
        bad = np.isin(np.round(t, 4), np.round(np.asarray(inv, float), 4))
        rel.append((dR, dp))
        okc.append(have[:-1] & have[1:] & ~bad[:-1] & ~bad[1:])
    si = {round(s, 4): i for i, s in enumerate(seq.scan_t)}
    clouds = {}

    def cloud(k):
        i = si.get(round(t[k], 4))
        if i is None:
            return None
        if k not in clouds:
            x = voxel_down(seq.scan(i), voxel)
            if len(x) > max_points:
                x = x[rng.choice(len(x), max_points, replace=False)]
            clouds[k] = x
        return clouds[k]

    n, C = len(t), len(cands)
    score = np.full((n - 1, C), -1.0)
    for k in range(n - 1):
        a, b = cloud(k), cloud(k + 1)
        if k > 0:
            clouds.pop(k - 1, None)
        if a is None or b is None:
            continue
        for c in range(C):
            if okc[c][k]:
                dR, dp = rel[c]
                score[k, c] = symmetric_overlap(a, dR[k].apply(b) + dp[k], thr)
    sc = score.copy()
    sc[:, 1:] -= margin
    pick = np.argmax(sc, 1)
    none = score.max(1) < 0                          # no valid candidate: keep cands[0], mark invalid
    pick[none] = 0
    R0, p0 = _T(cands[0][:1])
    Rs, ps = [R0[0]], [p0[0]]
    for k in range(n - 1):
        dR, dp = rel[pick[k]]
        ps.append(ps[-1] + Rs[-1].apply(dp[k]))
        Rs.append(Rs[-1] * dR[k])
    P = np.column_stack([t, np.array(ps), Rotation.concatenate(Rs).as_quat()])
    inv_t = t[1:][none | ~np.array([okc[pick[k]][k] for k in range(n - 1)])]
    info = {'picked': np.bincount(pick[~none], minlength=C).tolist(), 'no_candidate': int(none.sum()),
            'overlap_mean': score.max(1)[~none].mean().round(4).item() if (~none).any() else None,
            'overlap_gain_vs_0': float(np.mean(score.max(1)[~none] - np.maximum(score[~none, 0], 0)))
            if (~none).any() else None}
    return P, inv_t, info


def _align(P, t):
    """Rows of P at stamps t (nearest within 1 ms); returns (rows, have)."""
    j = np.searchsorted(P[:, 0], t).clip(1, len(P) - 1)
    j = np.where(np.abs(P[j - 1, 0] - t) < np.abs(P[j, 0] - t), j - 1, j)
    return P[j], np.abs(P[j, 0] - t) < 1e-3
