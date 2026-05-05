# Example: laying out a dataset for KISS-IMU

Suppose you have a KITTI-style sequence `07` and a DiTer-OS sequence
`Forest_new`. Place them under one shared `data_root`:

```
/storage1/For_IMUNet/
├── KITTI/
│   ├── 07/
│   │   ├── imu.csv
│   │   ├── gt_pose.csv          # 12-col SE(3), KITTI style
│   │   └── points/
│   │       ├── data/000000.bin … 004540.bin
│   │       └── timestamps.txt
│   └── 08/ …
└── DiTer_os/
    ├── Forest_new/
    │   ├── imu.csv
    │   ├── gt_pose.csv          # 7-col x,y,z,qx,qy,qz,qw
    │   └── points/
    │       ├── data/000000.bin … 012043.bin
    │       └── timestamps.txt
    ├── Lawn_lower_night/ …
    └── Park_in_day/ …
```

You then point `--data-root` at the appropriate parent and select
`--data-type` per dataset:

```bash
DATA_DIR=/storage1/For_IMUNet/KITTI    DATA_TYPE=kitti       TRAIN_SEQS="07"          bash scripts/train.sh
DATA_DIR=/storage1/For_IMUNet/DiTer_os DATA_TYPE=diter_os    TRAIN_SEQS="Forest_new"  bash scripts/train.sh
```

## Inspecting `imu.csv`

The first column must be a **monotonic timestamp in seconds**
(`float64`). The accelerometer / gyroscope columns are picked by
column index, not by name. Look at
[`src/data/seq_dataset.py`](../src/data/seq_dataset.py) for the
exact `acc_idx`/`gyr_idx` per `--data-type`. If your CSV has headers,
either strip them or load with `pandas.read_csv(..., header=None,
skiprows=1)` in a custom subclass.

## `points/data/*.bin`

Each `.bin` is a **packed numpy structured array** matching
`lidar_dtype` (per `--data-type`). For example,
`diter_os` uses

```python
np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<f8')])
```

and the loader does

```python
scan = np.fromfile(path, dtype=lidar_dtype)
points = np.stack([scan['x'], scan['y'], scan['z']], axis=-1)
```

If your scans are stored as float32 `(x, y, z, intensity)` (KITTI
convention), use `--data-type kitti`. If they are flat `(N, 4)` float
arrays of a custom dtype, write a subclass and override `load_scan`.

## `timestamps.txt`

One number per line, in seconds, in the same order as the sorted
`data/*.bin` files. The number of lines must match the number of bin
files (otherwise the loader trims to `min(len)`).

## Sanity-check before training

```python
from data.seq_dataset import SeqDataset
ds = SeqDataset('/storage1/For_IMUNet/DiTer_os', 'Forest_new', 'diter_os')
print(len(ds), ds[0].keys())
```

You should see a positive length and dict keys
`{scan0, scan1, imu_ts, accels, gyros, gt_pose0, gt_pose1, gt_velocity, ...}`.
