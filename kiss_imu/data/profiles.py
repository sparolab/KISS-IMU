"""Per-dataset layout and extrinsics.

A sequence directory holds

    imu.csv               t, then gyro / accelerometer columns (see gyr_col / acc_col)
    gt_pose.csv           t, x, y, z, qx, qy, qz, qw   (or t + a row-major 3x4 [R | t])
    points/data/*.bin     float64 x, y, z, intensity per point
    points/timestamps.txt one stamp per scan

Everything downstream works in the IMU frame:

    LiDAR point  -> IMU frame:  x_I = R_I_L x_L + t_I_L
    GT pose      -> world:      R = R_W_G R_gt, p = R_W_G p_gt  (R_W_G only re-orients the world;
                                 the GT body is taken to be the IMU)

Add a dataset by registering a profile here.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    gyr_col: int = 1                  # first gyro column in imu.csv
    acc_col: int = 4                  # first accelerometer column
    R_I_L: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_I_L: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R_W_G: np.ndarray = field(default_factory=lambda: np.eye(3))
    lidar_fields: int = 4             # float64 values per point in the .bin files


_FLIP_XY = np.diag([-1.0, -1.0, 1.0])

PROFILES = {
    'generic': DatasetProfile('generic'),
    'diter_os': DatasetProfile('diter_os'),
    'diter++': DatasetProfile('diter++', R_I_L=_FLIP_XY, R_W_G=_FLIP_XY),
    'botanic': DatasetProfile('botanic'),
    # GrandTour (converted export): GT and LiDAR already in the IMU frame
    'grandtour': DatasetProfile('grandtour'),
}


def get_profile(name):
    if name not in PROFILES:
        raise KeyError(f'unknown dataset type {name!r}; known: {sorted(PROFILES)}')
    return PROFILES[name]
