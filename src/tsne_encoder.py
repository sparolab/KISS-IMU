"""Visualize encoder features colored by GMM component (motion regime).

For each IMU window in the chosen sequences we:
  1. run the trained network's CNN+GRU encoder to get a (D, 128) feature map,
  2. pool it to a single 128-dim vector (mean over the time axis),
  3. classify the same window with the saved GMM (soft-mode reduce) to get
     a motion-regime label,
  4. run 2-D t-SNE on the collected features and save a PNG colored by the
     GMM component id.

If the encoder genuinely separates motion regimes, we expect clusters in
t-SNE space that correlate with GMM component colors.

The fitted GMM is expected to live next to the checkpoint as `gmm.joblib`
(train.py saves it there automatically). If you have an older run without
that file, copy in or symlink one.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# `src/` is on sys.path because we run from there (see tsne_encoder.sh).
from data.eval_dataset import SeqDataset, collate_fn
from models.imu_net import IMUNet
from models.gmm import GmmModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--data-type', type=str, required=True)
    p.add_argument('--ckpt',      type=str, required=True,
                   help='path to best_model.ckpt')
    p.add_argument('--gmm',       type=str, default=None,
                   help='path to gmm.joblib (defaults to <ckpt-dir>/gmm.joblib)')

    p.add_argument('--train-seqs', nargs='+', type=str, required=True,
                   help='sequences whose features go into the "train" plot')
    p.add_argument('--eval-seqs',  nargs='+', type=str, required=True,
                   help='sequences whose features go into the "eval" plot')

    p.add_argument('--out-dir',   type=str, required=True)
    p.add_argument('--device',    type=str, default='cuda:0')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=2)

    # t-SNE knobs — defaults are fine for a few-thousand-window run.
    p.add_argument('--perplexity', type=float, default=30.0)
    p.add_argument('--max-windows', type=int, default=5000,
                   help='cap on total points in the t-SNE plot (used to '
                        'derive a per-component cap when --per-comp is 0)')
    p.add_argument('--per-comp', type=int, default=0,
                   help='points per component in the stratified sample. '
                        '0 = auto (use min(component counts, max-windows / '
                        'K_eligible)). Set explicitly to force the same '
                        'count across components — clamped down per '
                        'component if a component has fewer points')
    p.add_argument('--min-per-comp', type=int, default=10,
                   help='components with fewer than this many windows in the '
                        'split are dropped from stratified sampling — they '
                        'have too few samples to form a meaningful cluster')
    p.add_argument('--pair-mode', type=str, default='all',
                   choices=['all', 'farthest'],
                   help='"all" plots every populated component. "farthest" '
                        'picks the two components whose GMM means are most '
                        'separated (Mahalanobis on the GMM\'s own scale) — '
                        'useful as a focused "does the encoder separate the '
                        'most-different motion regimes?" sanity check')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def farthest_pair(gmm: GmmModule) -> tuple[int, int]:
    """Return the two component ids whose GMM means are most separated
    under the GMM's own (whitened) feature scale.

    The GMM is fit in standardized [lin_speed, ang_speed] space, so plain
    Euclidean on `means_` is already a sensible distance. We weight further
    by the inverse covariance (squared Mahalanobis) so anisotropic clusters
    don't artificially inflate distance along their elongated axis.
    """
    M = gmm.gmm.means_                              # (K, 2)
    covs = gmm.gmm.covariances_                     # (K, 2, 2) for "full"
    K = M.shape[0]
    best = (0.0, 0, 1)
    for i in range(K):
        for j in range(i + 1, K):
            diff = M[i] - M[j]
            avg_cov = 0.5 * (covs[i] + covs[j])
            try:
                inv_cov = np.linalg.inv(avg_cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(avg_cov)
            d2 = float(diff @ inv_cov @ diff)
            if d2 > best[0]:
                best = (d2, i, j)
    return best[1], best[2]


@torch.no_grad()
def collect_features(network: IMUNet,
                     gmm: GmmModule,
                     seqs,
                     args) -> tuple[np.ndarray, np.ndarray]:
    """Iterate the given sequences; return (features, comp_ids).

    The encoder downsamples each window of length L into D feature vectors
    of size 128 (D = (L-10)/5+1 ish, set by the CNN stride). To match each
    feature to a GMM component we:
      1. predict GMM component for every raw IMU sample in the valid window
         (length L) — same features the GMM was fit on,
      2. split those L labels into D contiguous chunks (same split as
         IMUNet.broadcast_to_valid uses), majority-vote inside each chunk,
      3. emit one (feature, comp_id) pair per chunk → D rows per window.
    """
    feats_all, comps_all = [], []
    gravity_np = None

    for seq in seqs:
        ds = SeqDataset(data_root=args.data_root, data_seq=seq, data_type=args.data_type)
        dl = DataLoader(dataset=ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False,
                        drop_last=False, collate_fn=collate_fn)

        g = getattr(ds, "gravity", np.array([0., 0., 9.81], dtype=np.float32))
        gravity_np = g.cpu().numpy() if torch.is_tensor(g) else np.asarray(g, dtype=np.float64)

        for sample in dl:
            accels = sample['accels'].to(args.device).float()
            gyros  = sample['gyros'].to(args.device).float()
            valid_length = sample['valid_length']

            x = torch.cat([accels, gyros], dim=-1)        # (B, T, 6)
            feat = network.encoder(x, valid_length)       # (B, D, 128)
            feat_np = feat.detach().cpu().numpy()         # (B, D, 128)
            B, D, _ = feat_np.shape

            imu_ts_b = sample['imu_ts']                   # (B, T)
            accels_b = sample['accels']                   # (B, T, 3)
            gyros_b  = sample['gyros']                    # (B, T, 3)
            for b in range(B):
                L = int(valid_length[b])
                if L < 2:
                    continue

                lin_sp, ang_sp = gmm._imu_linear_angular_speed(
                    imu_ts_b[b, :L].cpu().numpy(),
                    accels_b[b, :L].cpu().numpy(),
                    gyros_b[b, :L].cpu().numpy(),
                    gravity_np,
                )
                X = np.stack([lin_sp, ang_sp], axis=1)    # (L, 2)
                labels = gmm.predict(X)                   # (L,)

                base = L // D
                rem  = L % D
                start = 0
                for d in range(D):
                    end = start + base + (1 if d < rem else 0)
                    end = min(end, L)
                    if end <= start:
                        break
                    chunk = labels[start:end]
                    if chunk.size == 0:
                        start = end
                        continue
                    cid = int(np.bincount(chunk, minlength=gmm.gmm.n_components).argmax())
                    feats_all.append(feat_np[b, d])
                    comps_all.append(cid)
                    start = end

    feats = np.stack(feats_all, axis=0) if feats_all else np.zeros((0, 128))
    comps = np.asarray(comps_all, dtype=np.int64)
    return feats, comps


def run_tsne(feats: np.ndarray, comps: np.ndarray, n_components: int,
             title: str, out_path: Path, args) -> None:
    if len(feats) == 0:
        print(f"[tsne_encoder] no features for {title} — skipping plot")
        return

    rng = np.random.default_rng(args.seed)

    populated = [k for k in range(n_components) if (comps == k).any()]
    counts = {k: int((comps == k).sum()) for k in populated}
    if not populated:
        print(f"[tsne_encoder] {title}: no populated components, skipping")
        return

    eligible = [k for k in populated if counts[k] >= args.min_per_comp]
    dropped  = [k for k in populated if counts[k] <  args.min_per_comp]
    if dropped:
        print(f"[tsne_encoder] {title}: dropping rare components "
              f"{ {k: counts[k] for k in dropped} } "
              f"(< --min-per-comp={args.min_per_comp})")
    if len(eligible) < 2:
        print(f"[tsne_encoder] {title}: only {len(eligible)} component(s) "
              f"have ≥{args.min_per_comp} samples — skipping stratified "
              f"sampling and falling back to all populated components")
        eligible = populated

    if args.per_comp > 0:
        per_comp_target = args.per_comp
        mode = "fixed"
    else:
        per_comp_target = min(
            min(counts[k] for k in eligible),
            max(1, args.max_windows // max(len(eligible), 1)),
        )
        mode = "auto"

    picked_idx = []
    per_comp_actual = {}
    for k in eligible:
        k_idx = np.flatnonzero(comps == k)
        n_take = min(per_comp_target, len(k_idx))
        chosen = rng.choice(k_idx, size=n_take, replace=False)
        picked_idx.append(chosen)
        per_comp_actual[k] = n_take
    picked_idx = np.concatenate(picked_idx)
    feats, comps = feats[picked_idx], comps[picked_idx]
    print(f"[tsne_encoder] {title}: stratified ({mode}, target="
          f"{per_comp_target}/component) × {len(eligible)} comps = "
          f"{len(feats)} points (taken/source: "
          f"{ {k: f'{per_comp_actual[k]}/{counts[k]}' for k in eligible} })")

    if len(feats) < 5:
        print(f"[tsne_encoder] {title}: only {len(feats)} samples — too few "
              f"for t-SNE, skipping")
        return

    perplexity = float(min(args.perplexity, max(2.0, (len(feats) - 1) / 3.0)))
    if perplexity >= len(feats):
        perplexity = max(2.0, len(feats) - 1.0)
    print(f"[tsne_encoder] {title}: running t-SNE on {feats.shape} features "
          f"(perplexity={perplexity:.1f})")
    emb = TSNE(n_components=2, perplexity=perplexity,
               random_state=args.seed, init='pca',
               learning_rate='auto').fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap('tab10', max(n_components, 1))
    for k in range(n_components):
        mask = comps == k
        n_k = int(mask.sum())
        if n_k == 0:
            ax.scatter([], [], s=8, color=cmap(k), label=f"comp {k} (n=0)")
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                   color=cmap(k), label=f"comp {k} (n={n_k})")
    ax.set_title(f"{title} — encoder features by GMM component (K={n_components})")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(loc='best', fontsize=9, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[tsne_encoder] {title}: saved → {out_path}")


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = Path(args.ckpt).resolve()
    gmm_path = Path(args.gmm).resolve() if args.gmm else ckpt_path.parent / "gmm.joblib"

    if not ckpt_path.exists():
        print(f"[tsne_encoder] ckpt not found: {ckpt_path}", file=sys.stderr)
        return 1
    if not gmm_path.exists():
        print(f"[tsne_encoder] GMM not found: {gmm_path}", file=sys.stderr)
        print("  Train with the updated train.py (it now writes gmm.joblib "
              "next to the run), or pass --gmm explicitly.", file=sys.stderr)
        return 1

    print(f"[tsne_encoder] ckpt: {ckpt_path}")
    print(f"[tsne_encoder] gmm:  {gmm_path}")

    network = IMUNet(prop_cov=True, device=args.device).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    network.load_state_dict(ckpt['network'] if 'network' in ckpt else ckpt)
    network.eval()

    # `train_packs` argument is required by __init__ but not used by load().
    gmm = GmmModule(train_packs=[]).load(str(gmm_path))
    K = int(gmm.gmm.n_components)
    print(f"[tsne_encoder] GMM has K={K} components")

    keep_set = None
    if args.pair_mode == 'farthest':
        a, b = farthest_pair(gmm)
        keep_set = {a, b}
        print(f"[tsne_encoder] pair-mode=farthest: keeping components {a},{b} "
              f"(GMM means: {gmm.gmm.means_[a]} vs {gmm.gmm.means_[b]})")

    for split_name, seqs in [("train", args.train_seqs), ("eval", args.eval_seqs)]:
        feats, comps = collect_features(network, gmm, seqs, args)
        if keep_set is not None and len(comps) > 0:
            mask = np.isin(comps, list(keep_set))
            feats, comps = feats[mask], comps[mask]
        out_path = Path(args.out_dir) / f"tsne_{split_name}.png"
        run_tsne(feats, comps, K, split_name, out_path, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
