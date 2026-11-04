"""Scenes: every animal of a session on one shared 30 Hz grid, so that an animal can see its neighbours.

Row k of every animal is the instant k / 30 s. `positions` are filtered as in the corpus, NaN outside a run. The
filter is symmetric, so a filtered position also holds up to 0.5 s of later motion, which a neighbour must not
reveal. So each scene also keeps what was observed by each instant, unfiltered: `seen`, the latest native
position at or before it, and `seen_step`, the native step that reached it, rescaled to one grid interval.
These are what an animal sees of its neighbours.
"""

import io
import json
import re
from pathlib import Path

import numpy as np

from . import corpus
from .corpus import RATE_HZ, sha256, write_json

FORMAT_VERSION = 1
SEED_ROWS = 2 * RATE_HZ  # a school's seed: 2 s of every animal
CONTINUATION_ROWS = 60 * RATE_HZ  # the real continuation a start must keep observed


def observed_by(run: corpus.Run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid rows inside the run's native interval, and at each its latest native position at or before
    that instant and the native step that reached it, per grid interval, NaN for the run's first frame."""
    native = corpus.to_common(run.xy, *run.frame_size)
    rows = corpus.grid_rows(run.t[0], run.t[-1])
    latest = np.searchsorted(run.t, rows / RATE_HZ + 1e-9, side="right") - 1
    step = np.full((len(rows), 2), np.nan)
    moved = latest > 0
    step[moved] = (native[latest[moved]] - native[latest[moved] - 1]) * run.native_hz / RATE_HZ
    return rows, native[latest], step


def build(runs: list[corpus.Run], records: list[dict], corpus_dir: Path, out: Path) -> dict:
    """One `<session>.npz` per session in out/ (positions, seen, seen_step, float32 [rows, animals, 2])
    and scenes.json, with each session's partition and hash. Split and scale are the corpus's."""
    manifest = corpus.partitions(corpus_dir)
    corpus_json = (corpus_dir / "corpus.json").read_bytes()
    by_session = {}
    for run in runs:
        by_session.setdefault(run.recording, []).append(run)
    out.mkdir(parents=True, exist_ok=True)
    sessions = {}
    for record in records:
        name = record["session"]
        steps = len(corpus.grid_rows(0.0, (record["frames"] - 1) / record["fps"]))
        positions = np.full((steps, record["animals"], 2), np.nan, dtype=np.float32)
        seen, seen_step = np.full_like(positions, np.nan), np.full_like(positions, np.nan)
        dropped = 0.0
        for run in by_session.get(name, []):
            animal = int(run.track) - 1
            rows, latest, step = observed_by(run)
            seen[rows, animal], seen_step[rows, animal] = latest, step
            placed = corpus.resample(run.t, corpus.to_common(run.xy, *run.frame_size), run.native_hz, shared_clock=True)
            if placed is None:
                dropped += float(run.t[-1] - run.t[0])
                continue
            grid, xy = placed
            first = round(grid[0] * RATE_HZ)
            positions[first : first + len(xy), animal] = xy
        observed = np.isfinite(positions[..., 0])
        others = (observed.sum(axis=1, keepdims=True) - 1) * observed  # neighbours seen by each observed animal
        buffer = io.BytesIO()
        np.savez(buffer, positions=positions, seen=seen, seen_step=seen_step)
        (out / f"{name}.npz").write_bytes(buffer.getvalue())
        sessions[name] = {
            "file": f"{name}.npz", "sha256": sha256(buffer.getvalue()),
            "group": record["group"], "partition": manifest[record["group"]],
            "animals": record["animals"], "steps": steps, "frame_size": record["frame_size"], "roi": record["roi"],
            "focal_hours": float(observed.sum() / RATE_HZ / 3600),
            "focal_hours_with_a_neighbour": float((others > 0).sum() / RATE_HZ / 3600),
            "neighbours_observed_mean": float(others[observed].mean()) if observed.any() else 0.0,
            "dropped_short_run_seconds": dropped,
        }  # fmt: skip
    totals = {}
    for s in sessions.values():
        t = totals.setdefault(s["partition"], {"sessions": 0, "focal_hours": 0.0, "focal_hours_with_a_neighbour": 0.0})
        t["sessions"] += 1
        t["focal_hours"] += s["focal_hours"]
        t["focal_hours_with_a_neighbour"] += s["focal_hours_with_a_neighbour"]
    summary = {
        "format_version": FORMAT_VERSION,
        "rate_hz": RATE_HZ,
        "coordinate_convention": corpus.CONVENTION,
        "scale": json.loads(corpus_json)["scale"],
        "corpus_sha256": sha256(corpus_json),
        "grid": "row k is time k / rate_hz from the session's first frame. positions: every run filtered as in the corpus and read only inside its filtered interval. seen: the latest native position at or before row k. seen_step: the native step that reached it, per grid interval",
        "partitions": totals,
        "sessions": sessions,
    }
    write_json(out / "scenes.json", summary)
    return summary


def read(scenes_dir: Path) -> dict:
    """scenes.json: the scale, and each session's entry by name."""
    return json.loads((scenes_dir / "scenes.json").read_text(encoding="utf-8"))


def sessions(scenes_dir: Path, partition: str) -> dict[str, dict]:
    """The `partition` sessions' entries in scenes.json, by name."""
    return {s: e for s, e in read(scenes_dir)["sessions"].items() if e["partition"] == partition}


def load(scenes_dir: Path, entry: dict, *names: str) -> list[np.ndarray]:
    """The named arrays [rows, animals, 2] of a session's scenes.json entry, NaN where unobserved, checked
    against the entry's hash."""
    return corpus.load_npz(scenes_dir / entry["file"], entry["sha256"], *names)


def arena(entry: dict, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The lower and upper corners, in the common frame, of the box around the session's tracking
    region (an idtracker.ai ellipse or polygon), or without one, around its positions."""
    if not entry.get("roi"):
        return np.nanmin(positions.reshape(-1, 2), axis=0), np.nanmax(positions.reshape(-1, 2), axis=0)
    (roi,) = entry["roi"]
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", roi)]
    if "Polygon" in roi:
        corners = np.array(numbers).reshape(-1, 2)
    elif "Ellipse" in roi:
        cx, cy, ax, ay = numbers[:4]
        corners = np.array([[cx - ax, cy - ay], [cx + ax, cy + ay]])
    else:
        raise ValueError(f"unknown region {roi}")
    common = corpus.to_common(corners, *entry["frame_size"])
    return common.min(0), common.max(0)


def starts(positions: np.ndarray, count: int) -> list[int]:
    """`count` start rows spread over the session, each with every animal observed through its seed and
    at least 90% of animal-time observed over the next CONTINUATION_ROWS."""
    full = np.isfinite(positions[..., 0])
    every = np.concatenate([[0], np.cumsum(full.all(1))])  # prefix sums: rows with every animal, observed cells
    cells = np.concatenate([[0], np.cumsum(full.sum(1))])
    s = np.arange(SEED_ROWS, len(positions) - CONTINUATION_ROWS - 1)
    seeded = every[s + 1] - every[s - SEED_ROWS] == SEED_ROWS + 1
    ok = s[seeded & ((cells[s + CONTINUATION_ROWS + 1] - cells[s]) / ((CONTINUATION_ROWS + 1) * full.shape[1]) >= 0.9)].tolist()
    if len(ok) < count:
        raise ValueError(f"only {len(ok)} usable starts")
    return [ok[i] for i in np.linspace(0, len(ok) - 1, count).round().astype(int)]
