"""The corpus: every observed run filtered, resampled to 30 Hz in one coordinate frame, and split by group.

Positions are centred on the frame, x right and y up, divided by the frame's longer side. A token is the
displacement between consecutive 30 Hz positions divided by the corpus scale, the RMS training displacement.
"""

import functools
import hashlib
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

FORMAT_VERSION = 1
PREPROCESSING_VERSION = 2
RATE_HZ = 30
CONVENTION = "centered_x_right_y_up_max_extent"
SPLIT_SEED = 20261014
HELD_OUT_SHARE = 0.1  # of a source's groups for each held-out partition, and the most duration one group may hold

# Low-pass filter, applied to every run before linear interpolation: a 33-tap Hamming-window FIR with unit gain
# at DC and a cutoff of min(12 Hz, 0.4 x native rate). Without it, interpolating 29.1 fps tracking jitter onto the
# 30 Hz grid keeps the jitter where grid points meet samples and averages it between, a false 0.9 Hz swell in
# speed. `valid` convolution drops 16 native samples at each end of a run.
FIR_TAPS = 33
FIR_CUTOFF_HZ = 12.0
FIR_CUTOFF_FRACTION = 0.4
FIR_HALF = (FIR_TAPS - 1) // 2


@dataclass
class Run:
    """Timed observations of one animal with nothing unobserved inside."""

    source: str
    recording: str
    track: str
    group: str
    t: np.ndarray  # float64 seconds, uniform spacing 1/native_hz
    xy: np.ndarray  # float64 [n, 2] pixels, origin top left, y down
    native_hz: float
    frame_size: tuple[int, int]  # width, height of the full frame
    provenance: dict = field(default_factory=dict)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(dest: Path, data, indent: int | None = 1) -> None:
    """JSON in UTF-8 with Unix line ends, the same bytes on every system."""
    dest.write_text(json.dumps(data, indent=indent), encoding="utf-8", newline="\n")


def to_common(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Full-frame pixels to x right, y up, centred, divided by the longer side."""
    extent = max(width, height)
    return np.column_stack([(xy[:, 0] - width / 2) / extent, (height / 2 - xy[:, 1]) / extent])


@functools.cache
def taps(native_hz: float) -> np.ndarray:
    return signal.firwin(FIR_TAPS, min(FIR_CUTOFF_HZ, FIR_CUTOFF_FRACTION * native_hz), fs=native_hz)


def resample(t: np.ndarray, xy: np.ndarray, native_hz: float, shared_clock: bool = False):
    """Times and positions on the 30 Hz grid that ends at the run's last filtered time, or None if
    fewer than three fit or the run is shorter than the filter. Never extrapolates.

    With shared_clock, the grid is instead the times k / RATE_HZ inside the filtered interval, so runs
    of different animals of one session share their grid times (the scenes)."""
    if np.abs(np.diff(t) - 1 / native_hz).max(initial=0) > 1e-9 + 1e-6 / native_hz:
        raise ValueError("resample expects uniformly sampled observations")
    if len(t) < FIR_TAPS or (t[-1 - FIR_HALF] - t[FIR_HALF]) * RATE_HZ + 1e-9 < 2:
        return None
    xy = np.column_stack([np.convolve(xy[:, i], taps(native_hz), mode="valid") for i in range(2)])
    t = t[FIR_HALF : len(t) - FIR_HALF]
    if shared_clock:
        grid = np.arange(np.ceil(t[0] * RATE_HZ - 1e-9), np.floor(t[-1] * RATE_HZ + 1e-9) + 1) / RATE_HZ
        if len(grid) < 3:
            return None
    else:
        count = int(np.floor((t[-1] - t[0]) * RATE_HZ + 1e-9)) + 1
        grid = t[-1] - np.arange(count - 1, -1, -1) / RATE_HZ
    return grid, np.column_stack([np.interp(grid, t, xy[:, i]) for i in range(2)])


def split_groups(seconds_by_source: dict[str, dict[str, float]]) -> dict[str, dict[str, str]]:
    """Seeded shuffle of each source's groups into validation and test, each about 10% of the groups
    (at least one), and train. A held-out group holds at most 10% of the source's duration, so the
    largest sessions stay in training. A held-out partition that nothing fits gets the shortest
    training group. Each source has its own generator, so adding a source never changes another's split."""
    manifest = {}
    for source in sorted(seconds_by_source):
        seconds = seconds_by_source[source]
        if len(seconds) < 3:
            raise ValueError(f"{source}: {len(seconds)} independent groups, at least 3 needed")
        rng = np.random.default_rng([SPLIT_SEED, zlib.crc32(source.encode())])
        groups = sorted(seconds)
        quota = max(1, round(HELD_OUT_SHARE * len(groups)))
        room = HELD_OUT_SHARE * sum(seconds.values())
        assignment = {}
        for group in (groups[i] for i in rng.permutation(len(groups))):
            open_ = [p for p in ("validation", "test") if list(assignment.values()).count(p) < quota]
            assignment[group] = open_[0] if open_ and seconds[group] <= room else "train"
        for partition in ("validation", "test"):
            if partition not in assignment.values():
                shortest = min((g for g, p in assignment.items() if p == "train"), key=seconds.get)
                assignment[shortest] = partition
        manifest[source] = dict(sorted(assignment.items()))
    return manifest


def build(runs: list[Run], out: Path) -> dict:
    """Resample every run, freeze the group splits, and write train/validation/test.npz, corpus.json,
    sequences.jsonl and split-manifest.json to out/."""
    seconds_by_source = {}
    for run in runs:
        groups = seconds_by_source.setdefault(run.source, {})
        groups[run.group] = groups.get(run.group, 0.0) + float(run.t[-1] - run.t[0])
    manifest = split_groups(seconds_by_source)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "split-manifest.json"
    write_json(manifest_path, {"seed": SPLIT_SEED, "groups": manifest})

    parts = {"train": [], "validation": [], "test": []}
    rows = {"train": [], "validation": [], "test": []}
    dropped = {"runs": 0, "seconds": 0.0}
    for run in runs:
        resampled = resample(run.t, to_common(run.xy, *run.frame_size), run.native_hz)
        if resampled is None:
            dropped["runs"] += 1
            dropped["seconds"] += float(run.t[-1] - run.t[0])
            continue
        positions = resampled[1]
        partition = manifest[run.source][run.group]
        rows[partition].append({
            "partition": partition, "index": len(parts[partition]), "source": run.source,
            "recording": run.recording, "track": run.track, "group": run.group,
            "native_interval_s": [float(run.t[0]), float(run.t[-1])], "positions": len(positions),
            **run.provenance,
        })  # fmt: skip
        parts[partition].append(positions.astype(np.float32))

    train = np.concatenate([np.diff(p.astype(np.float64), axis=0) for p in parts["train"]])
    scale = float(np.sqrt(np.mean((train**2).sum(axis=1) / 2)))
    arrays = {}
    for partition, sequences in parts.items():
        lengths = [len(p) for p in sequences]
        path = out / f"{partition}.npz"
        np.savez(path, positions=np.concatenate(sequences), sequence_offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64))
        arrays[partition] = {"file": path.name, "sha256": sha256(path.read_bytes()), "sequences": len(lengths), "positions": int(sum(lengths))}
    with (out / "sequences.jsonl").open("w", newline="\n") as f:
        for partition in parts:
            for row in rows[partition]:
                f.write(json.dumps(row) + "\n")
    # The field names and texts are those of the corpora the shipped models were trained on, whose hashes
    # the exported model.json records.
    corpus = {
        "format_version": FORMAT_VERSION,
        "rate_hz": RATE_HZ,
        "coordinate_convention": CONVENTION,
        "scale": scale,
        "preprocessing_version": PREPROCESSING_VERSION,
        "filter": {"design": "scipy.signal.firwin, hamming", "taps": FIR_TAPS, "cutoff_hz": FIR_CUTOFF_HZ,
                   "cutoff": "min(cutoff_hz, 0.4 x native rate)", "applied": "to every run, before linear interpolation", "edge": "valid convolution, (taps-1)/2 native samples trimmed per end"},
        "arrays": arrays,
        "split_manifest_sha256": sha256(manifest_path.read_bytes()),
        "dropped_short_runs": dropped,
    }  # fmt: skip
    write_json(out / "corpus.json", corpus)
    return corpus


def load_npz(path: Path, expected_sha256: str, *names: str) -> list[np.ndarray]:
    """The named arrays of an .npz file, without pickle, after checking its hash."""
    if sha256(path.read_bytes()) != expected_sha256:
        raise ValueError(f"{path} does not match the hash recorded for it")
    with np.load(path, allow_pickle=False) as data:
        return [data[name] for name in names]


def partitions(corpus_dir: Path) -> dict[str, str]:
    """Every source's split groups, {group: partition}."""
    families = json.loads((corpus_dir / "split-manifest.json").read_text())["groups"]
    return {group: partition for groups in families.values() for group, partition in groups.items()}


def sequence_field(corpus_dir: Path, partition: str, name: str) -> list:
    """One field of each of a partition's sequences, in the order of `load`'s offsets."""
    rows = (json.loads(line) for line in (corpus_dir / "sequences.jsonl").read_text(encoding="utf-8").splitlines())
    return [r[name] for r in rows if r["partition"] == partition]


def load(corpus_dir: Path, partition: str) -> tuple[np.ndarray, np.ndarray, float]:
    """A partition's positions, sequence offsets and the training scale."""
    corpus = json.loads((corpus_dir / "corpus.json").read_text())
    entry = corpus["arrays"][partition]
    positions, offsets = load_npz(corpus_dir / entry["file"], entry["sha256"], "positions", "sequence_offsets")
    return positions, offsets, corpus["scale"]
