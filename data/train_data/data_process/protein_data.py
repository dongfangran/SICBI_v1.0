"""Apply a state-distance cutoff to stat.py caches and write protein NPY files."""

import argparse
from pathlib import Path

import numpy as np


def finalize(source, destination, cutoff):
    candidates = np.load(source / "candidate_input.npy")
    populations = np.load(source / "populations.npy")
    distances = np.load(source / "state_distance.npy")
    count = len(populations)
    if distances.shape != (count, count) or count == 0 or populations.sum() <= 0:
        raise ValueError(f"Invalid candidate cache: {source}")

    # Largest clusters become representatives; candidate -> retained is directional.
    retained, assigned = [], {}
    for candidate in sorted(range(count), key=lambda i: (-populations[i], i)):
        close = [i for i in retained if np.isfinite(distances[candidate, i])
                 and distances[candidate, i] < cutoff]
        if close:
            assigned[candidate] = min(close, key=lambda i: distances[candidate, i])
        else:
            retained.append(candidate)
            assigned[candidate] = candidate

    rows, probabilities = [], []
    total = float(populations.sum())
    for new_id, original in enumerate(retained, start=1):
        features = candidates[candidates[:, 0] == original + 1, 1:]
        if not len(features):
            raise ValueError(f"State {original + 1} has no features in {source}")
        rows.append(np.column_stack((np.full(len(features), new_id), features)))
        population = sum(populations[i] for i in range(count) if assigned[i] == original)
        probabilities.append((new_id, population / total))

    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / "input.npy", np.concatenate(rows).astype(np.float32))
    np.save(destination / "state_probabilities.npy",
            np.asarray(probabilities, dtype=np.float32))
    return len(retained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("stat_cache"))
    parser.add_argument("--output", type=Path, default=Path("protein_output"))
    parser.add_argument("--cutoff", type=float, required=True,
                        help="States closer than this are merged")
    args = parser.parse_args()
    if not 0 <= args.cutoff <= 2:
        parser.error("--cutoff must be between 0 and 2")
    sources = [path.parent for path in sorted(
        args.cache.glob("shard_*/*/state_distance.npy")
    )]
    if not sources:
        parser.error(f"No shard caches found under {args.cache}")
    ids = [source.name for source in sources]
    if len(ids) != len(set(ids)):
        parser.error("Duplicate PDB IDs across shard caches")

    for source in sources:
        count = finalize(source, args.output / "proteins" / source.name, args.cutoff)
        print(f"{source.name}: {count} states")


if __name__ == "__main__":
    main()
