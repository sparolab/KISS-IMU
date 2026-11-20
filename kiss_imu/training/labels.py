"""Pseudo-labels from a PGO run's per-scan log (no GT).

Shared by tools/make_pseudo_labels.py (from steps.jsonl on disk) and the warm-start runner (from the
in-memory log of a running PGO, scripts/run_warmstart.py).

    source 'smoothed'  the PGO poses (fused with the IMU); pass them as `poses`
    source 'lo'        the raw LO scan-to-map poses of every step (independent of the IMU between scans; the
                       PGO must log them: log.lo_pose: true). Quality comes from the rotation block of the LO
                       information.

A scan is invalid when its LO was rejected or its quality is below min_eig_frac x the median of the accepted
scans. quality = eigenvalue / median (1 = typical, small = weakly constrained).

"""

import numpy as np


def labels_from_steps(steps, source='lo', min_eig_frac=0.05, poses=None):
    """steps: list of per-scan dicts (SlidingWindowPGO.log). poses: (n, 8) TUM rows for source 'smoothed'.
    Returns dict(P (n, 8) TUM label poses, t (m,) step stamps, invalid_t, quality (m, 2) [stamp, q])."""
    if source == 'lo':
        steps = [s for s in steps if s.get('lo_pose') is not None]
        if not steps:
            raise ValueError("no LO poses in the log: run the PGO with log.lo_pose: true")
        P = np.array([[s['t'], *s['lo_pose']] for s in steps])
    elif source == 'smoothed':
        if poses is None:
            raise ValueError("source 'smoothed' needs the smoothed poses")
        P = np.asarray(poses, float)
    else:
        raise ValueError(f'unknown label source {source!r} (lo | smoothed)')
    st = np.array([s['t'] for s in steps])
    acc = np.array([bool(s.get('lo_accepted', False)) for s in steps])
    if source == 'lo':                   # rotation-block information: how well the scan pins the attitude
        eig = np.array([np.linalg.eigvalsh(np.reshape(s['lo_info'], (6, 6))[:3, :3])[0] if s.get('lo_info') else np.nan
                        for s in steps])
    else:
        eig = np.array([s.get('lo_h_min_eig', np.nan) or np.nan for s in steps], float)
    med = np.nanmedian(eig[acc]) if acc.any() else np.nan
    with np.errstate(invalid='ignore'):
        bad = (~acc) | ~(eig > min_eig_frac * med) if min_eig_frac else ~acc
    return dict(P=P, t=st, invalid_t=st[bad], quality=np.column_stack([st, eig / med]),
                invalid_frac=float(bad.mean()) if len(bad) else float('nan'))
