<div align="center">
  <h1>
    KISS-IMU: Self-supervised Inertial Odometry <br> 
    with Motion-balanced Learning and Uncertainty-aware Inference</h1>
  <a href="https://github.com/sparolab"><img src="https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54" /></a>
  <a href="https://sparolab.github.io/research/kiss_imu/"><img src="https://github.com/sparolab/Joint_ID/blob/main/fig/badges/badge-website.svg" alt="Project" /></a>
  <a href="https://arxiv.org/abs/2603.06205"><img src="https://img.shields.io/badge/arXiv-2603.06205-b31b1b.svg?style=flat-square" alt="arXiv" /></a>
  <a href="https://www.youtube.com/watch?v=cjAFROi-jG0"><img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" /></a>
  <br />

<h3>🏆 [IEEE ICRA 2026 Award Finalist]</h3>

  <a href="https://scholar.google.com/citations?user=wL8VdUMAAAAJ&hl=ko" target="_blank">Jiwon Choi</a><sup></sup>,
  <a href="https://hogyun2.github.io/" target="_blank">Hogyun Kim</a><sup></sup>,
  <a href="https://scholar.google.com/citations?user=kiBTkqMAAAAJ&hl=ko" target="_blank">Geonmo Yang</a><sup></sup>,
  <a href="https://scholar.google.com/citations?user=4-5Fi9kAAAAJ&hl=ko" target="_blank">Juhui Lee</a><sup></sup>,
  <a href="https://scholar.google.com/citations?user=W5MOKWIAAAAJ&hl=ko" target="_blank">Younggun Cho</a><sup>†</sup>

**[🤖 Spatial AI and Robotics Lab (SPARO)](https://sites.google.com/view/sparo/%ED%99%88?authuser=0&pli=1)**

  <p align="center"><img src="fig/main.gif" alt="animated" width="75%" /></p>

</div>

---

## 📰 News
- 🏆 **[May 6, 2026]** KISS-IMU is selected as an **IEEE ICRA 2026 Award Finalist**!
- 🎉 **[Jan 31, 2026]** KISS-IMU is **accepted to IEEE ICRA 2026**.

## 💡 What is KISS-IMU?
KISS-IMU learns to denoise raw IMU streams against a self-generated
LiDAR-odometry pseudo-label, with a GMM-based motion-balanced sampler and
a frequency gate so under-represented motion regimes are not drowned out
during training.

## 🚀 How to use KISS-IMU?

### 🐳 Quick start with Docker

The base image (`sparolab/kiss-imu:v1.0`) already contains all
runtime deps (CUDA, PyTorch, pypose, kiss-icp, small_gicp, pygicp,
scikit-learn, …). The compose file mounts the repo as the working
directory automatically — you only need to point one line at your
dataset path.

```bash
$ git clone https://github.com/sparolab/KISS-IMU.git
$ cd KISS-IMU

# 1) edit docker/docker-compose.yml — replace `{dataset_folder}` in
#    the `volumes:` section with the absolute path to your datasets
#    (e.g. /mnt/hdd/datasets:/storage1)

# 2) (optional, only if you need GUI apps like RViz from inside)
$ xhost +local:root

# 3) launch
$ docker compose -f docker/docker-compose.yml up -d
$ docker exec -it kiss-imu-ws bash

# inside the container — pwd is already /home/test_ws/src
$ bash scripts/train.sh
```

### 🛠️ Without Docker

```bash
$ pip install -r requirements.txt
$ bash scripts/train.sh
```

### 📂 Dataset layout

KISS-IMU expects one top-level `data_root` with one sub-directory per
sequence:

```
📁 <data_root>/
└── 📂 <SEQUENCE_NAME>/
    ├── 📄 imu.csv
    ├── 📄 gt_pose.csv
    └── 📂 points/
        ├── 📂 data/
        │   ├── 🟦 000000.bin
        │   └── ...
        └── 📄 timestamps.txt
```

Pick the matching `--data-type` (e.g. `kitti`, `mulran`, `diter_os`)
and the loader handles dataset-specific column indices and coordinate
transforms. For custom formats or detailed schemas, see
[examples/dataset_layout.md](examples/dataset_layout.md).

### 🏋️ Training

```bash
# default: DiTer-OS / Forest_new with KISS-ICP backend
$ bash scripts/train.sh

# point at a different dataset
$ DATA_DIR=/storage/KITTI DATA_TYPE=kitti TRAIN_SEQS="07" VALID_SEQS="07" \
  LO_MODEL=kiss_icp bash scripts/train.sh
```

All hyper-params are tunable via env vars at the top of
[`scripts/train.sh`](scripts/train.sh). To reproduce the **raw-IMU + PVGO**
baseline (denoted `†` in the paper), use:

```bash
$ bash scripts/raw_pvgo.sh
```

### 📊 Evaluation

```bash
# (A) evaluate one specific checkpoint
$ CKPT=results/.../ckpt/0017.ckpt \
  EVAL_SEQS="Forest_new Lawn_lower_night Park_in_day" \
  bash scripts/evaluate.sh

# (B) sweep a directory of checkpoints and pick the best
$ CKPT_DIR=results/.../ckpt \
  EVAL_SEQS="Forest_new Lawn_lower_night Park_in_day" \
  SELECT_METRIC=balanced \
  bash scripts/evaluate.sh
```

Per-window endpoint RPE (translation, rotation) and last-step APE are
averaged over each evaluation sequence. With multiple checkpoints, the
best one is selected by `--select-metric` — either a single metric
(`ape`, `rpe_trans`, `rpe_rot`) or `balanced` (normalized combination
of all three). Run `bash scripts/evaluate.sh --help` for the full set
of options.

## 🔗 Supplementary
- 📄 [arXiv](https://arxiv.org/abs/2603.06205)
- 🌐 [Project page](https://sparolab.github.io/research/kiss_imu/)
- 🎬 [Video](https://www.youtube.com/watch?v=cjAFROi-jG0)

## 📝 Citation
If you find this work useful, please consider citing:

```bibtex
@inproceedings{choi2026kissimu,
  title     = {KISS-IMU: Self-supervised Inertial Odometry with
               Motion-balanced Learning and Uncertainty-aware Inference},
  author    = {Choi, Jiwon and Kim, Hogyun and Yang, Geonmo and Lee, Juhui and Cho, Younggun},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

## 📬 Contact
- Jiwon Choi — 📧 jiwon2@inha.edu

## 📜 License
For academic usage, the code is released under the BSD 3.0 license. For any commercial purpose, please contact the authors.

## ✨ Contributors
<a href="https://github.com/sparolab/KISS-IMU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=sparolab/KISS-IMU" />
</a>
