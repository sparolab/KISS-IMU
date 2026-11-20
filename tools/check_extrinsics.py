"""Sanity check of a sequence's frames and clocks as the loader delivers them (everything should be in the IMU
frame, on the IMU clock):

  A. LiDAR vs gyro   scan-to-scan GICP (scans k, k+S) relative rotations against the gyro integrated over the same
                     interval, for clock shifts dt on a grid: R (Procrustes of the rotation vectors) and dt at the
                     smallest residual. Points already in the IMU frame and on its clock: R ~ I, dt ~ 0.
  B. GT vs gyro      the same with the GT relative rotations (GT with orientation only).
  C. gravity         first second's mean specific force, rotated by the GT attitude, against world +z.
Only intervals turning > 2 deg count; gyro bias = median over the first second (the platform stands still).

    python3 tools/check_extrinsics.py <config> <seq> [...] [--pairs 400]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import small_gicp
from scipy.spatial.transform import Rotation as R

from kiss_imu.config import load_config
from kiss_imu.data import Sequence


def gyro_rel(t, w, t0, t1):
    """Integrated gyro rotation over [t0, t1] (vectorised over intervals; IMU samples treated as piecewise constant)."""
    out = []
    for a, b in zip(t0, t1):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b)
        if j - i < 2:
            out.append(np.full(3, np.nan)); continue
        ts = np.r_[a, t[i:j], b]
        ws = np.vstack([w[max(i - 1, 0)], w[i:j]])
        q = R.identity()
        for k in range(len(ts) - 1):
            q = q * R.from_rotvec(ws[k] * (ts[k + 1] - ts[k]))
        out.append(q.as_rotvec())
    return np.array(out)


def procrustes(A, B):
    """R minimising sum |A - R B|^2 (rows), residual deg (RMS)."""
    H = B.T @ A
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    res = np.degrees(np.sqrt(np.mean(np.sum((A - B @ Rm.T) ** 2, 1))))
    return R.from_matrix(Rm), res


def fit(t0, t1, rv, seq, w, grid):
    """For each clock shift: R and a residual gyro bias b jointly (gyro_int - b (t1 - t0) ~ R target, alternating
    Procrustes and least squares for b), as a hand-held start is not still enough for the first-second bias."""
    best = None
    dur = (t1 - t0)[:, None]
    for dt in grid:
        g = gyro_rel(seq.imu_t, w, t0 + dt, t1 + dt)
        ok = np.isfinite(g).all(1)
        b = np.zeros(3)
        for _ in range(5):
            Rx, res = procrustes(g[ok] - b * dur[ok], rv[ok])
            b = np.linalg.lstsq(np.repeat(dur[ok], 1, 0), g[ok] - rv[ok] @ Rx.as_matrix().T, rcond=None)[0][0]
        if best is None or res < best[2]:
            best = (dt, Rx, res, ok.sum(), b)
    return best


def fit_pooled(items, grid):
    """items: [(t0, t1, rv, seq, w)]; one R and clock shift for all, a residual bias per sequence."""
    best = None
    for dt in grid:
        G = [gyro_rel(s.imu_t, w, t0 + dt, t1 + dt) for t0, t1, rv, s, w in items]
        oks = [np.isfinite(g).all(1) for g in G]
        bs = [np.zeros(3) for _ in items]
        for _ in range(5):
            A = np.vstack([g[o] - b * (it[1] - it[0])[o][:, None] for g, o, b, it in zip(G, oks, bs, items)])
            Bm = np.vstack([it[2][o] for o, it in zip(oks, items)])
            Rx, res = procrustes(A, Bm)
            bs = [np.linalg.lstsq((it[1] - it[0])[o][:, None], g[o] - it[2][o] @ Rx.as_matrix().T, rcond=None)[0][0]
                  for g, o, it in zip(G, oks, items)]
        if best is None or res < best[2]:
            best = (dt, Rx, res, int(sum(o.sum() for o in oks)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('seqs', nargs='+')
    ap.add_argument('--pairs', type=int, default=400); ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--tmax', type=float, default=None, help='only scans within tmax s of the start (A)')
    ap.add_argument('--pool', action='store_true', help='also one LiDAR->IMU fit over all given sequences')
    a = ap.parse_args()
    pool = []
    d = load_config(a.config)['data']
    for s in a.seqs:
        seq = Sequence(d['root'], s, d.get('type', 'generic'), (d.get('gt_time_offset') or {}).get(s, 0.0),
                       min_range=d.get('min_range', 1.0), max_range=d.get('max_range', 100.0), imu_dir=d.get('imu_dir'),
                       imu_ext=d.get('imu_ext', False), gt_file=d.get('gt_file', 'gt_pose.csv'),
                       scan_time_offset=d.get('scan_time_offset', 0.0), lidar_rotvec_deg=d.get('lidar_rotvec_deg'))
        t = seq.imu_t
        bg = np.median(seq.gyr[t < t[0] + 1.0], 0)
        w = seq.gyr - bg
        print(f'== {s}: imu {seq.imu_rate:.0f} Hz, |bias| {np.degrees(np.linalg.norm(bg)):.3f} deg/s, scans {len(seq.scan_t)}', flush=True)
        # A. LiDAR
        if len(seq.scan_t) > a.step + 10:
            ns = len(seq.scan_t) if a.tmax is None else int(np.searchsorted(seq.scan_t, seq.scan_t[0] + a.tmax))
            idx = np.linspace(0, ns - a.step - 1, min(4 * a.pairs, ns - a.step - 1)).astype(int)
            t0, t1 = seq.scan_t[idx], seq.scan_t[idx + a.step]
            g0 = gyro_rel(t, w, t0, t1)
            keep = np.where(np.degrees(np.linalg.norm(g0, axis=1)) > 2.0)[0][:a.pairs]
            rv, kk = [], []
            for i in keep:
                pa, pb = seq.scan(idx[i]), seq.scan(idx[i] + a.step)
                pa, pb = pa[:, :3], pb[:, :3]
                r = small_gicp.align(pa, pb, registration_type='GICP', downsampling_resolution=0.3,
                                     max_correspondence_distance=1.0, num_threads=4, max_iterations=30)
                if r.converged:
                    rv.append(R.from_matrix(r.T_target_source[:3, :3]).as_rotvec()); kk.append(i)
            rv, kk = np.array(rv), np.array(kk)
            if len(kk) > 20:
                dt, Rx, res, n, b = fit(t0[kk], t1[kk], rv, seq, w, np.arange(-0.1, 0.1001, 0.005))
                dt2, Rx, res, n, b = fit(t0[kk], t1[kk], rv, seq, w, np.arange(dt - 0.006, dt + 0.0061, 0.001))
                pool.append((t0[kk], t1[kk], rv, seq, w))
                sv = np.linalg.svd(rv, compute_uv=False)
                print(f'  A axis spread (singular values of the rotation vectors, rel.): {(sv / sv[0]).round(2)}')
                print(f'  A LiDAR->IMU   rot {np.degrees(Rx.magnitude()):5.2f} deg  rotvec_deg {np.degrees(Rx.as_rotvec()).round(2)}'
                      f'  dt {dt2 * 1e3:+5.1f} ms  resid {res:.2f} deg  n {n}  extra bias {np.degrees(np.linalg.norm(b)):.2f} deg/s', flush=True)
            else:
                print('  A LiDAR: too few turning pairs')
        # B / C. GT orientation
        if seq.has_gt:
            q = R.from_matrix(seq.gt_R)
            has_rot = np.degrees((q[0].inv() * q).magnitude()).max() > 1.0
            if not has_rot:
                print('  B GT: position only (no orientation)')
                continue
            tg = np.arange(seq.gt_t[0] + 1.0, seq.gt_t[-1] - 1.5, 0.37)
            ta, tb = tg, tg + 0.5
            Ga, Gb = seq.gt_states(ta)[0], seq.gt_states(tb)[0]
            rv = R.from_matrix(np.einsum('nji,njk->nik', Ga, Gb)).as_rotvec()
            m = np.degrees(np.linalg.norm(rv, axis=1)) > 2.0
            sel = np.where(m)[0][:a.pairs * 2]
            dt, Rx, res, n, b = fit(ta[sel], tb[sel], rv[sel], seq, w, np.arange(-0.1, 0.1001, 0.005))
            dt2, Rx, res, n, b = fit(ta[sel], tb[sel], rv[sel], seq, w, np.arange(dt - 0.006, dt + 0.0061, 0.001))
            sv = np.linalg.svd(rv[sel], compute_uv=False)
            print(f'  B axis spread {(sv / sv[0]).round(2)}')
            print(f'  B GT body->IMU rot {np.degrees(Rx.magnitude()):5.2f} deg  rotvec_deg {np.degrees(Rx.as_rotvec()).round(2)}'
                  f'  dt {dt2 * 1e3:+5.1f} ms  resid {res:.2f} deg  n {n}  extra bias {np.degrees(np.linalg.norm(b)):.2f} deg/s', flush=True)
            k = (t >= seq.gt_t[0]) & (t < seq.gt_t[0] + 1.0)
            f = seq.acc[k].mean(0)
            fw = seq.gt_states(np.array([t[k].mean()]))[0][0] @ f
            tilt = np.degrees(np.arccos(fw[2] / np.linalg.norm(fw)))
            print(f'  C gravity      |f| {np.linalg.norm(f):.3f} m/s2, tilt vs world z {tilt:.2f} deg', flush=True)

    if a.pool and len(pool) > 1:
        dt, Rx, res, n = fit_pooled(pool, np.arange(-0.05, 0.0501, 0.005))
        dt, Rx, res, n = fit_pooled(pool, np.arange(dt - 0.004, dt + 0.0041, 0.001))
        print(f'== pooled LiDAR->IMU over {len(pool)} seqs: rot {np.degrees(Rx.magnitude()):.2f} deg  rotvec_deg '
              f'{np.degrees(Rx.as_rotvec()).round(3).tolist()}  dt {dt * 1e3:+.1f} ms  resid {res:.2f} deg  n {n}')


if __name__ == '__main__':
    main()
