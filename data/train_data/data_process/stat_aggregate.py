"""Combine completed stat.py shards and draw one state-distance histogram."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("stat_cache"))
    parser.add_argument("--output", type=Path, default=Path("stat_cache/summary"))
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--hist-max", type=float, default=0.35,
                        help="Upper edge of the plotted distance range")
    args = parser.parse_args()
    if args.bins < 1 or args.hist_max <= 0:
        parser.error("--bins and --hist-max must be positive")

    paths = sorted(args.cache_root.glob("shard_*/*/state_distance.npy"))
    if not paths:
        parser.error(f"No completed shard distances found in {args.cache_root}")

    seen, values = set(), []
    for path in paths:
        pdb_id = path.parent.name
        if pdb_id in seen:
            raise ValueError(f"Duplicate complex across shards: {pdb_id}")
        seen.add(pdb_id)
        expected = ("candidate_input.npy", "populations.npy",
                    "representative_frames.npy")
        if not all((path.parent / name).is_file() for name in expected):
            raise ValueError(f"Incomplete candidate cache: {path.parent}")
        matrix = np.load(path, allow_pickle=False)
        if matrix.shape != (10, 10) or np.isnan(matrix).any():
            raise ValueError(f"Invalid distance matrix: {path}")
        mask = np.isfinite(matrix) & ~np.eye(10, dtype=bool)
        values.append(matrix[mask])

    distances = np.concatenate(values).astype(np.float32)
    if not len(distances):
        raise ValueError("No finite off-diagonal state distances were found")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "all_state_distances.npy", distances)
    plt.hist(distances, bins=args.bins, range=(0, args.hist_max))
    plt.xlabel("One-way state distance (mean 1 - cosine)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(args.output / "state_distance_hist.png", dpi=180)
    plt.close()
    print(f"Combined {len(seen)} complexes and {len(distances)} distances")


if __name__ == "__main__":
    main()
