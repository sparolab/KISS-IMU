"""Run the sliding-window LiDAR-inertial PGO on sequences and score them against GT.

    python3 scripts/run_pgo.py configs/diter_os/pgo.yaml --seqs Forest_new --out runs/raw
    python3 scripts/run_pgo.py configs/diter_os/pgo.yaml --seqs Forest_new --out runs/corr \
        --set io.mode=corrected io.checkpoint=runs/warmstart/Forest_new/train_r2/last.pt
    # live view in Rerun (spawns the viewer), or record to a file and open it later with `rerun <file>.rrd`
    python3 scripts/run_pgo.py configs/diter_os/pgo.yaml --seqs Forest_new --out runs/viz --rerun spawn
    python3 scripts/run_pgo.py configs/diter_os/pgo.yaml --seqs Forest_new --out runs/viz --rerun save
"""

import argparse
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # tiny dense solves; BLAS threads only fight small_gicp's

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from kiss_imu.config import load_config  # noqa: E402
from kiss_imu.data import Sequence  # noqa: E402
from kiss_imu.eval.metrics import evaluate  # noqa: E402
from kiss_imu.pgo.pipeline import run_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rerun', choices=['spawn', 'save', 'connect'], help='Rerun visualisation sink')
    ap.add_argument('--rerun-addr', default=None, help="for 'connect': viewer gRPC url (default: local viewer)")
    ap.add_argument('--rerun-scan-points', type=int, default=10000, help='points of the current scan shown')
    ap.add_argument('--set', nargs='*', default=[], dest='overrides', metavar='KEY=VALUE', help='dotted config overrides')
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    d = cfg['data']
    summary = {}
    for name in a.seqs:
        print(f'== {name}  (io.mode={cfg["io"]["mode"]}, registration={cfg["lo"]["registration"]["name"]}, '
              f'map={cfg["lo"]["map"]["name"]})', flush=True)
        seq = Sequence(d['root'], name, d.get('type', 'generic'), (d.get('gt_time_offset') or {}).get(name, 0.0),
                       min_range=d.get('min_range', 1.0),
                       max_range=d.get('max_range', 100.0), imu_dir=d.get('imu_dir'), despike=d.get('imu_despike'), imu_ext=d.get('imu_ext', False), gt_file=d.get('gt_file', 'gt_pose.csv'),
                       scan_time_offset=d.get('scan_time_offset', 0.0), lidar_rotvec_deg=d.get('lidar_rotvec_deg'))
        out = os.path.join(a.out, name)
        observers = []
        if a.rerun:
            from kiss_imu.viz import RerunViz
            path = os.path.join(out, 'viz.rrd') if a.rerun == 'save' else a.rerun_addr
            os.makedirs(out, exist_ok=True)
            observers.append(RerunViz(seq, a.rerun, path, scan_points=a.rerun_scan_points))
        pgo = run_sequence(cfg, seq, out, observers=observers)
        res = {k: evaluate(np.array(getattr(pgo, k)), seq) for k in ('online', 'smoothed')}
        res['lo_accept_rate'] = float(np.mean([x['lo'] for x in pgo.log]))
        res['ms_per_scan'] = float(np.mean([x['dt_ms'] for x in pgo.log]))
        summary[name] = res
        with open(os.path.join(out, 'result.json'), 'w') as f:
            json.dump(res, f, indent=1)
        o = res['online']
        print(f'   ATE {o["ate_rmse"]:.3f} m (aligned {o["ate_aligned_rmse"]:.3f})  RPE {o["rpe_pct"]:.2f} %  '
              f'rot {o["rot_rmse_deg"]:.2f} deg  over {o["length_m"]:.0f} m  | LO accepted {res["lo_accept_rate"]:.1%}  '
              f'{res["ms_per_scan"]:.0f} ms/scan', flush=True)
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
