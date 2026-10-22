"""Build 10 candidate protein states for one range of MISATO complexes."""

import argparse
import gzip
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import parmed as pmd
import torch
from netCDF4 import Dataset
from scipy.spatial import cKDTree


# Edit these defaults for one shard, or override them on the command line.
DEFAULT_START_INDEX = 12000
DEFAULT_END_INDEX = None  # Exclusive; use None for the final open-ended shard.
DEFAULT_CACHE = Path("stat_cache/shard_12000_final")


def optional_end_index(value):
    """Accept an integer or 'none' for the final open-ended shard."""
    return None if value.lower() == "none" else int(value)


try:
    from model import dMaSIF
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Cannot import dMaSIF: missing '{error.name}'. Place model.py and its "
        "geometry_processing.py, helper.py and benchmark_models.py dependencies "
        "on PYTHONPATH, and install their Python packages."
    ) from error


RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN",
    "GLU", "GLH", "GLY", "HID", "HIE", "HIP", "HIS", "ILE", "LEU",
    "LYS", "LYN", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}
ELEMENT_CHANNEL = {6: 0, 1: 1, 8: 2, 7: 3, 16: 4, 34: 5}
# MISATO stores element *categories*, rather than atomic numbers, in
# ``atoms_element``.  The categories used by standard protein atoms are
# H=1, C=2, N=3, O=4 and S=7.  Keep the topology's atomic numbers for all
# downstream chemistry so dMaSIF receives its expected element channels.
MISATO_ELEMENT_TO_ATOMIC_NUMBER = {1: 1, 2: 6, 3: 7, 4: 8, 7: 16}


class SkipComplex(ValueError):
    """Expected data exclusion, as opposed to a pipeline failure."""


def load_dmasif(checkpoint, device):
    # Keep these settings identical to build_dataset.py and its pretrained weights.
    args = SimpleNamespace(
        atom_dims=6, curvature_scales=[1., 2., 3., 5., 10.], distance=1.05,
        dropout=0., emb_dims=16, embedding_layer="dMaSIF", in_channels=16,
        k=40, n_layers=3, no_chem=False, no_geom=False, orientation_units=16,
        post_units=8, radius=12., resolution=1., search=True, site=False,
        single_protein=True, sup_sampling=20, use_mesh=False, device=device,
    )
    net = dMaSIF(args).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
    net.eval()
    return net


def atom_mask(indices):
    """Compact zero-based atom indices into a cpptraj one-based mask."""
    values = sorted(set(int(i) + 1 for i in indices))
    if not values:
        raise SkipComplex("empty cpptraj atom mask")
    spans = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            spans.append(f"{start}-{end}" if start != end else str(start))
            start = end = value
    spans.append(f"{start}-{end}" if start != end else str(start))
    return "@" + ",".join(spans)


def read_complex(group, topology_path, pocket_radius):
    coordinates = group["trajectory_coordinates"]
    n_frames, n_atoms, xyz_dim = coordinates.shape
    if xyz_dim != 3 or n_frames < 10:
        raise SkipComplex("invalid trajectory shape or fewer than 10 frames")
    residue_codes = group["atoms_residue"][:]
    element_codes = group["atoms_element"][:].astype(int)
    starts = group["molecules_begin_atom_index"][:].astype(int)
    ligand = np.flatnonzero(residue_codes == 0)
    if not len(ligand):
        raise SkipComplex("peptide/no small-molecule ligand (atoms_residue == 0 absent)")
    if not len(starts) or not 0 < starts[-1] < n_atoms:
        raise SkipComplex("invalid molecule boundary")
    if not np.array_equal(ligand, np.arange(starts[-1], n_atoms)):
        raise SkipComplex("ligand atoms do not match the last molecule")
    if len(element_codes) != n_atoms:
        raise SkipComplex("atom metadata length mismatch")

    full = pmd.load_file(str(topology_path))
    if len(full.atoms) < n_atoms:
        raise SkipComplex("topology has fewer atoms than HDF5")
    # MISATO HDF5 contains the protein/ligand prefix; solvent follows in .top.
    top = full[:n_atoms]
    top.box = None
    atomic_numbers = np.asarray([a.atomic_number for a in top.atoms], dtype=int)
    expected_atomic_numbers = np.asarray([
        MISATO_ELEMENT_TO_ATOMIC_NUMBER.get(int(code), 0)
        for code in element_codes
    ], dtype=int)
    mapped = expected_atomic_numbers > 0
    if np.any(atomic_numbers[mapped] != expected_atomic_numbers[mapped]):
        raise SkipComplex("HDF5/topology atom order or element mismatch")
    if any(top.atoms[i].residue.name.strip().upper() in RESIDUES for i in ligand):
        raise SkipComplex("ligand is protein-like/peptide")

    reference = np.asarray(coordinates[0], dtype=np.float32)
    ligand_heavy = ligand[atomic_numbers[ligand] > 1]
    if not len(ligand_heavy):
        raise SkipComplex("ligand has no heavy atoms")
    center = reference[ligand_heavy].mean(axis=0)
    protein = np.arange(starts[-1])
    protein = np.asarray([
        i for i in protein if top.atoms[i].residue.name.strip().upper() in RESIDUES
    ], dtype=int)
    protein_heavy = protein[atomic_numbers[protein] > 1]
    near = protein_heavy[
        np.linalg.norm(reference[protein_heavy] - center, axis=1) <= pocket_radius
    ]
    pocket_residues = {top.atoms[i].residue.idx for i in near}
    pocket = np.asarray([
        i for i in protein if top.atoms[i].residue.idx in pocket_residues
    ], dtype=int)
    pocket_heavy = pocket[atomic_numbers[pocket] > 1]
    if not len(pocket_heavy):
        raise SkipComplex("empty 12 A protein pocket")
    backbone = np.asarray([
        i for i in pocket_heavy if top.atoms[i].name.strip().upper() in {"N", "CA", "C"}
    ], dtype=int)
    fit_atoms = backbone if len(backbone) >= 3 else pocket_heavy
    joint = np.concatenate((pocket_heavy, ligand_heavy))
    return (coordinates, top, atomic_numbers, protein, pocket_heavy, fit_atoms,
            joint, center)


def write_netcdf(path, coordinates):
    n_frames, n_atoms, _ = coordinates.shape
    with Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as nc:
        nc.Conventions = "AMBER"
        nc.ConventionVersion = "1.0"
        nc.program = "stat.py"
        nc.programVersion = "1.0"
        nc.createDimension("frame", None)
        nc.createDimension("atom", n_atoms)
        nc.createDimension("spatial", 3)
        xyz = nc.createVariable("coordinates", "f4", ("frame", "atom", "spatial"))
        xyz.units = "angstrom"
        for frame in range(n_frames):
            xyz[frame] = np.asarray(coordinates[frame], dtype=np.float32)


def cluster_frames(work, top, coordinates, fit_atoms, joint, cpptraj):
    top.save(str(work / "complex.prmtop"), format="amber", overwrite=True)
    write_netcdf(work / "trajectory.nc", coordinates)
    script = "\n".join([
        "parm complex.prmtop",
        "trajin trajectory.nc",
        f"rms first {atom_mask(fit_atoms)}",
        f"cluster C1 hieragglo clusters 10 averagelinkage rms {atom_mask(joint)} nofit "
        "out assignments.dat summary clusters.dat",
        "trajout aligned.nc netcdf",
        "run", "",
    ])
    (work / "cluster.in").write_text(script, encoding="ascii")
    with (work / "cpptraj.log").open("w", encoding="utf-8") as log:
        subprocess.run([cpptraj, "-i", "cluster.in"], cwd=work,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = np.atleast_2d(np.loadtxt(work / "assignments.dat", comments="#"))
    if len(rows) != coordinates.shape[0] or rows.shape[1] < 2:
        raise RuntimeError("cpptraj did not assign every frame")
    labels = rows[:, 1].astype(int)
    unique = np.unique(labels)
    if len(unique) != 10:
        raise RuntimeError(f"expected 10 cpptraj clusters, obtained {len(unique)}")
    with Dataset(work / "aligned.nc") as nc:
        aligned = np.asarray(nc.variables["coordinates"][:], dtype=np.float32)
    return labels, unique, aligned


def medoid_frame(aligned, frames, joint):
    # The cluster medoid is a real frame, avoiding dependence on cpptraj's frame numbering.
    x = aligned[frames][:, joint, :].reshape(len(frames), -1).astype(np.float64)
    squared = np.sum(x * x, axis=1)
    distances = np.maximum(squared[:, None] + squared[None, :] - 2 * x @ x.T, 0)
    return int(frames[np.argmin(distances.sum(axis=1))])


def protein_patch(frame, elements, protein, ligand_center, net, device, pocket_radius, seed):
    # dMaSIF samples pseudo-surface points stochastically; make cache creation reproducible.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    protein = np.asarray([i for i in protein if int(elements[i]) in ELEMENT_CHANNEL])
    atoms = torch.as_tensor(frame[protein], dtype=torch.float32, device=device)
    types = torch.zeros((len(protein), 6), dtype=torch.float32, device=device)
    channels = torch.as_tensor([ELEMENT_CHANNEL[int(elements[i])] for i in protein],
                               device=device)
    types[torch.arange(len(protein), device=device), channels] = 1
    surface = {
        "atoms": atoms, "atom_xyz": atoms, "atomtypes": types,
        "batch_atoms": torch.zeros(len(protein), dtype=torch.long, device=device),
        "mesh_labels": None,
    }
    with torch.no_grad():
        net.preprocess_surface(surface)
        result = net(surface)["P1"]
    xyz = result["xyz"].detach().cpu().numpy()
    embedding = result["embedding_1"].detach().cpu().numpy()
    # The descriptor pocket uses the same ligand-center 12 A definition as the
    # residue pocket; unlike the RMSD selection it retains surface points directly.
    keep = np.linalg.norm(xyz - ligand_center, axis=1) <= pocket_radius
    if not keep.any():
        raise SkipComplex("dMaSIF found no pocket surface points")
    return xyz[keep].astype(np.float32), embedding[keep].astype(np.float32)


def state_distance(a, b, match_radius, min_match):
    xyz_a, emb_a = a
    xyz_b, emb_b = b
    _, nearest = cKDTree(xyz_b).query(xyz_a, distance_upper_bound=match_radius)
    matched = nearest < len(xyz_b)
    if matched.mean() < min_match:
        return np.inf
    x = emb_a[matched]
    y = emb_b[nearest[matched]]
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    valid = (nx > 1e-8) & (ny > 1e-8)
    if valid.mean() * matched.mean() < min_match:
        return np.inf
    cosine = np.sum((x[valid] / nx[valid, None]) *
                    (y[valid] / ny[valid, None]), axis=1)
    return float(np.mean(1 - np.clip(cosine, -1, 1)))


def process(group, top_path, work, args, net):
    coords, top, elements, protein, pocket, fit, joint, ligand_center = read_complex(
        group, top_path, args.pocket_radius
    )
    labels, clusters, aligned = cluster_frames(work, top, coords, fit, joint, args.cpptraj)
    patches, populations, representatives = [], [], []
    for cluster in clusters:
        frames = np.flatnonzero(labels == cluster)
        representative = medoid_frame(aligned, frames, joint)
        patches.append(protein_patch(
            aligned[representative], elements, protein, ligand_center, net, args.device,
            args.pocket_radius, args.seed + 1009 * int(representative) + len(patches)
        ))
        populations.append(len(frames))
        representatives.append(representative)
    distances = np.full((10, 10), np.inf, dtype=np.float32)
    for i in range(10):
        distances[i, i] = 0
        for j in range(10):
            if i != j:
                distances[i, j] = state_distance(
                    patches[i], patches[j], args.match_radius, args.min_match
                )
    rows = [np.column_stack((np.full(len(features), i + 1), features))
            for i, (_, features) in enumerate(patches)]
    return (np.concatenate(rows).astype(np.float32),
            np.asarray(populations, dtype=np.int32), distances,
            np.asarray(representatives, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=Path("data/MD.hdf5"))
    parser.add_argument("--archive", type=Path,
                        default=Path("data/parameter_restart_files_MD.tar.gz"))
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX,
                        help="First eligible topology entry, zero-based and inclusive")
    parser.add_argument("--end-index", type=optional_end_index,
                        default=DEFAULT_END_INDEX,
                        help="Stop before this eligible topology entry; 'none' means EOF")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/dMaSIF_search_3layer_12A_16dim"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--pocket-radius", type=float, default=12.)
    parser.add_argument("--match-radius", type=float, default=2.)
    parser.add_argument("--min-match", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed for reproducible dMaSIF surface sampling")
    args = parser.parse_args()
    if args.start_index < 0 or (args.end_index is not None
                                and args.end_index <= args.start_index):
        parser.error("require 0 <= --start-index < --end-index")
    if args.pocket_radius <= 0 or args.match_radius <= 0:
        parser.error("all radius parameters must be positive")
    if not 0 < args.min_match <= 1:
        parser.error("--min-match must be in (0, 1]")
    # This model.py calls torch.cuda.synchronize() unconditionally during forward().
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("the supplied model.py requires an available CUDA device")
    args.cache.mkdir(parents=True, exist_ok=True)
    net = load_dmasif(args.checkpoint, args.device)
    skipped, completed, reused, eligible_count = [], 0, 0, 0

    with h5py.File(args.h5, "r") as h5, tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("/production.top.gz"):
                continue
            pdb_id = Path(member.name).parent.name
            if not re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id):
                continue
            index = eligible_count
            eligible_count += 1
            if args.end_index is not None and index >= args.end_index:
                break
            if index < args.start_index:
                continue
            key = pdb_id.upper() if pdb_id.upper() in h5 else pdb_id.lower()
            if key not in h5:
                skipped.append((pdb_id, "missing from HDF5"))
                continue
            target = args.cache / key.upper()
            if target.exists():
                expected = ("candidate_input.npy", "populations.npy",
                            "state_distance.npy", "representative_frames.npy")
                if not target.is_dir() or not all((target / name).is_file()
                                                  for name in expected):
                    raise RuntimeError(f"Incomplete cache; inspect before resuming: {target}")
                reused += 1
                print(f"[cached] index={index} {key}", flush=True)
                continue
            try:
                with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
                    work = Path(temporary)
                    # tar.gz and production.top.gz are both decompressed only for this ID.
                    with gzip.GzipFile(fileobj=archive.extractfile(member)) as src, \
                            (work / "production.top").open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    result = process(h5[key], work / "production.top", work, args, net)
                    staging = work / "cache_entry"
                    staging.mkdir()
                    for name, array in zip(("candidate_input.npy", "populations.npy",
                                            "state_distance.npy", "representative_frames.npy"), result):
                        np.save(staging / name, array)
                    staging.rename(target)
                completed += 1
                print(f"[new] index={index} {key}", flush=True)
            except SkipComplex as error:
                skipped.append((key, str(error)))
                print(f"[skip] {key}: {error}", flush=True)

    (args.cache / "skipped.tsv").write_text(
        "pdb_id\treason\n" + "".join(f"{key}\t{reason}\n" for key, reason in skipped),
        encoding="utf-8",
    )
    print(f"Shard [{args.start_index}, {args.end_index}) in {args.cache}: "
          f"{completed} new, {reused} cached, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
