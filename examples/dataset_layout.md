# Example: laying out a dataset for KISS-IMU

`diter_os` is the reference dataset. Place each sequence under one shared
`data_root`:

```
/storage1/Datasets/kiss_imu_datasets/DiTer_os/
├── Forest_new/
│   ├── imu.csv
│   ├── gt_pose.csv          # 8-col: timestamp + x,y,z,qx,qy,qz,qw
│   └── points/
│       ├── data/<timestamp>.bin …   # any names; sorted lexically
│       └── timestamps.txt
├── Lawn_lower_night/ …
└── Park_in_day/ …
```

Point `--data-root` at the parent and select `--data-type`:

```bash
DATA_DIR=/storage1/Datasets/kiss_imu_datasets/DiTer_os DATA_TYPE=diter_os TRAIN_SEQS="Forest_new" bash scripts/train.sh
```

The `.bin` files can be named anything (sequential numbers or nanosecond
timestamps) — `load_scan` just sorts them lexically and zips them with
`timestamps.txt` line-by-line, so the sort order and the timestamp order
must agree.

## Adding your own dataset

Each `--data-type` maps to one branch in `SeqDataset.__init__`
([`src/data/seq_dataset.py`](../src/data/seq_dataset.py)) that sets the
LiDAR dtype, the IMU column indices, and the extrinsics. To onboard a new
dataset, add a branch (or edit the generic `else` fallback) with:

- **`lidar_dtype`** — only change it if your `.bin` layout differs from the
  default `(x, y, z, intensity)` float64 structured array.
- **`acc_idx` / `gyr_idx`** — the column offsets of the accel/gyro triplets
  in your `imu.csv`.
- **`R_I_L` / `R_I_G`** — your IMU↔LiDAR and IMU↔global extrinsic rotations.

## `imu.csv`

The first column must be a **monotonic timestamp in seconds** (`float64`).
The accelerometer / gyroscope columns are picked by index (`acc_idx`,
`gyr_idx`), not by name. If your CSV has headers, strip them or load with
`pandas.read_csv(..., header=None, skiprows=1)` in a custom subclass.

## `gt_pose.csv`

The first column is the timestamp; the rest is the pose. Two pose formats
are supported:

- **7-col quaternion** — `x, y, z, qx, qy, qz, qw` (8 columns total with
  the leading timestamp).
- **SE(3) row-major** — the 9 rotation + 3 translation entries, i.e. a
  12-col pose (13 columns total with the leading timestamp).

## `points/data/*.bin`

Each `.bin` is a **packed numpy structured array** matching `lidar_dtype`.
The default (`diter_os` and the generic fallback) is a 4-field float64
dtype:

```python
np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<f8')])
```

and the loader does

```python
scan = np.fromfile(path, dtype=lidar_dtype)
points = np.vstack((scan['x'], scan['y'], scan['z'])).T   # (N, 3)
```

If your scans use a different layout, set `lidar_dtype` accordingly in your
`--data-type` branch, or write a subclass and override `load_scan`.

## `timestamps.txt`

One number per line, in seconds, in the same order as the sorted
`data/*.bin` files. The number of lines must match the number of bin
files (otherwise the loader trims to `min(len)`).

## Sanity-check before training

```python
from data.seq_dataset import SeqDataset
ds = SeqDataset('/storage1/Datasets/kiss_imu_datasets/DiTer_os', 'Forest_new', 'diter_os')
print(len(ds), ds[0].keys())
```

You should see a positive length and dict keys
`{scan0, scan1, imu_ts, accels, gyros, gt_pose0, gt_pose1, gt_velocity, ...}`.
