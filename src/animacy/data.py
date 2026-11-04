"""The idtracker.ai recordings: download from the authors' public Google Drive folder, checks, and runs.

Each session folder holds `trajectories/with_gaps_csv/trajectories.csv` (the tracker's centroids, empty where an
animal was not identified) and `trajectories/validated_csv/` (after human validation, complete). A position is
used only where the tracker measured it and validation kept it unchanged.
"""

import csv
import dataclasses
import functools
import html
import json
import re
import struct
import time
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np

from .corpus import Run, sha256

TRACKED_SESSIONS = "1VH97_bNFz09Ke_kBL1oV2HbTS25BwnKC"  # the "tracked_sessions" folder linked from idtracker.ai
VIDEOS = "1oGpHsu2EfWqU-mSMhO3iAtEKCpcoPIT4"  # its videos, of which only each AVI header is read
FOLDER_URL = "https://drive.google.com/embeddedfolderview?id={}"
FILE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
ENTRY = re.compile(r'flip-entry" id="entry-([^"]+)".*?flip-entry-title">([^<]*)<', re.S)
USER_AGENT = "animacy (github.com/gustavecortal/animacy)"
CHUNK = 4 << 20  # bytes per range request, so a dropped connection costs little

SESSION_FILES = {
    "session.json": "session.json",
    "with_gaps.csv": "trajectories/with_gaps_csv/trajectories.csv",
    "validated.csv": "trajectories/validated_csv/trajectories.csv",
    "validated_attributes.json": "trajectories/validated_csv/attributes.json",
}

# Split groups: recordings that may film the same animals stay in one partition. Several "zebrafish"
# sessions are medaka. The 80-fish videos have exactly the frame size of one LargeGroups setup each.
FISH = {
    "zebrafish_60_1": "largegroups_1",
    "zebrafish_60_2": "largegroups_2",
    "zebrafish_100_2": "largegroups_2",
    "zebrafish_60_3": "largegroups_3",
    "zebrafish_100_3": "largegroups_3",
    "zebrafish_10_1": "tu20171214_group4",
    "zebrafish_10_2": "tu20171214_group5",
    "zebrafish_10_3": "tu20171214_group6",
    "zebrafish_80_1": "largegroups_1",
    "zebrafish_80_2": "largegroups_2",
    "zebrafish_80_3": "largegroups_3",
    "zebrafish_10_4": "medaka_2012may31",
    "zebrafish_5": "medaka_2012may31",
    "zebrafish_20": "medaka_20fish_2013",
    "zebrafish_2": "nacre_pair3_2013",
    "zebrafish_7": "wik8_grupo4_2014",
    "zebrafish_8": "conflict_2012",
}
# Fly sessions whose videos share a frame size may film the same animals, so they share a group.
FLIES = {
    "drosophila_10": "flies_1280x960",
    "drosophila_100_1": "flies_3712",
    "drosophila_100_3": "flies_3712",
    "drosophila_38": "flies_3456x3400",
    "drosophila_72": "flies_3456x3400",
    "drosophila_59": "flies_2432x2400",
    "drosophila_6": "flies_804x808",
    "drosophila_8": "flies_1024",
}
FAMILIES = {"fish": FISH, "flies": FLIES}
REFERENCE_BODY = 0.0202  # the median fish body length, in frame units, which body-length positions keep



def pinned() -> dict[str, dict[str, str]]:
    """The SHA-256 of every file the models were trained on, by session."""
    return json.loads(resources.files(__package__).joinpath("sessions.json").read_text(encoding="utf-8"))


def _get(url: str, start: int, end: int, attempts: int = 8) -> tuple[bytes, int]:
    """Bytes start..end inclusive (fewer at the end of the file) and the file's size. Drive sometimes
    answers with a web page instead of the file, which is retried like a dropped connection, unless it
    says the file's download quota is used up."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    if b"Quota exceeded" in response.read():
                        raise RuntimeError(f"{url}: Google Drive's download quota for this file is used up. Run fetch again later: it keeps the files it has.")
                    raise ConnectionError(f"{url}: a web page instead of the file")
                if response.status == 206:
                    return response.read(), int(response.headers["Content-Range"].rpartition("/")[2])
                if response.status == 200 and start == 0:  # Drive sends some small files whole
                    data = response.read(end + 1)
                    return data, int(response.headers.get("Content-Length") or len(data))
                raise ValueError(f"{url}: HTTP {response.status} to a range request")
        except OSError:  # including HTTPError and ConnectionError
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def get_range(url: str, start: int = 0, end: int | None = None) -> bytes:
    """Bytes start..end inclusive, or to the end of the file, in requests of CHUNK bytes."""
    parts = []
    while end is None or start <= end:
        data, total = _get(url, start, start + CHUNK - 1 if end is None else min(start + CHUNK - 1, end))
        if not data and start < total:
            raise ValueError(f"{url}: empty response at byte {start} of {total}")
        parts.append(data)
        start += len(data)
        end = total - 1 if end is None else min(end, total - 1)
    return b"".join(parts)


def write_atomic(dest: Path, data: bytes) -> None:
    """Write through a temporary file, so an interrupted download never leaves a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    partial.write_bytes(data)
    partial.replace(dest)


@functools.cache
def list_folder(folder_id: str) -> dict[str, str]:
    """Names and IDs of a public Drive folder's entries."""
    request = urllib.request.Request(FOLDER_URL.format(folder_id), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode()
    return {html.unescape(name): entry for entry, name in ENTRY.findall(page)}


def find(folder_id: str, path: str) -> str:
    """The ID of a file or folder at a slash-separated path below a Drive folder."""
    for part in path.split("/"):
        entries = list_folder(folder_id)
        if part not in entries:
            raise FileNotFoundError(f"{part} not in Drive folder {folder_id}")
        folder_id = entries[part]
    return folder_id


def video_header(session: str) -> dict:
    """Frame size, frame rate and length from the first 64 KiB of the session's AVI."""
    data = get_range(FILE_URL.format(list_folder(VIDEOS)[f"{session}.avi"]), 0, 65535)
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise ValueError(f"{session}.avi is not an AVI file")
    i = data.find(b"avih")
    width, height = struct.unpack_from("<2I", data, i + 40)
    # A stream header is "strh", its size, then the stream type: the video stream's gives the rate.
    i = next(i for i in range(len(data) - 12) if data[i : i + 4] == b"strh" and data[i + 8 : i + 12] == b"vids")
    scale, rate = struct.unpack_from("<2I", data, i + 28)
    (length,) = struct.unpack_from("<I", data, i + 40)
    return {"size": [width, height], "rate": rate, "scale": scale, "fps": rate / scale, "frames": length}


def fetch(session: str, data_dir: Path) -> None:
    """Download one session's files into data_dir/<session>/ and check them against sessions.json."""
    folder = data_dir / session
    expected = pinned()[session]
    remote = None
    for name in expected:
        path = folder / name
        if path.exists() and sha256(path.read_bytes()) == expected[name]:
            continue
        remote = remote or find(TRACKED_SESSIONS, f"session_{session}")
        if name == "inconsistent_frames.csv":
            data = get_range(FILE_URL.format(list_folder(remote)[name]))
        elif name == "video_header.json":
            data = json.dumps(video_header(session)).encode()
        else:
            data = get_range(FILE_URL.format(find(remote, SESSION_FILES[name])))
        if sha256(data) != expected[name]:
            raise ValueError(f"{session}/{name} differs from the file the models were trained on")
        write_atomic(path, data)


def load_positions(path: Path) -> np.ndarray:
    """A trajectories CSV (`x1, y1, x2, y2, ...` in pixels) as float64 [frames, animals, 2], NaN where empty."""
    with path.open(newline="") as f:
        rows = csv.reader(f)
        header = next(rows)
        if len(header) % 2 or header[:2] != ["x1", "y1"]:
            raise ValueError(f"{path}: unexpected header {header[:5]}")
        values = np.array([[float(x) if x else np.nan for x in row] for row in rows])
    return values.reshape(len(values), -1, 2)


def observed(with_gaps: np.ndarray, validated: np.ndarray, excluded_frames: list[int]) -> np.ndarray:
    """[frames, animals] True where the tracker measured a centroid and validation kept it. Frames where
    the tracker found more blobs than animals are left out."""
    mask = np.isfinite(with_gaps).all(-1) & (np.abs(with_gaps - validated).max(-1) == 0)
    mask[excluded_frames] = False
    return mask


def runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """(animal, first frame, last frame) of each run of at least two consecutive True frames."""
    out = []
    for animal in range(mask.shape[1]):
        m = mask[:, animal]
        edges = np.r_[0, np.flatnonzero(m[1:] != m[:-1]) + 1, len(m)]
        out += [(animal, int(a), int(b - 1)) for a, b in zip(edges[:-1], edges[1:]) if m[a] and b - a >= 2]
    return out


def session(name: str, group: str, data_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """A session's record (group, animals, frames, fps, frame_size, roi, inconsistent_frames), its
    measured positions and their observed mask, after checking the files agree with the video."""
    folder = data_dir / name
    expected = pinned()[name]
    for file in expected:
        if sha256((folder / file).read_bytes()) != expected[file]:
            raise ValueError(f"{folder / file} differs from the file the models were trained on: run fetch again")
    meta = json.loads((folder / "session.json").read_text())
    header = json.loads((folder / "video_header.json").read_text())
    with_gaps = load_positions(folder / "with_gaps.csv")
    validated = load_positions(folder / "validated.csv")
    if with_gaps.shape != validated.shape or len(with_gaps) != header["frames"]:
        raise ValueError(f"{name}: exports {with_gaps.shape} and {validated.shape}, video {header['frames']} frames")
    if header["size"] != [meta["original_width"], meta["original_height"]] or meta["resolution_reduction"] != 1:
        raise ValueError(f"{name}: video {header['size']}, session {meta['original_width']}x{meta['original_height']}")
    inconsistent = folder / "inconsistent_frames.csv"
    excluded = [int(x) for x in inconsistent.read_text().split()] if inconsistent.exists() else []
    record = {
        "session": name, "group": group, "animals": with_gaps.shape[1], "frames": len(with_gaps),
        "fps": header["fps"], "frame_size": header["size"], "roi": meta["roi_list"], "inconsistent_frames": excluded,
    }  # fmt: skip
    return record, with_gaps, observed(with_gaps, validated, excluded)


def family_runs(family: str, data_dir: Path) -> tuple[list[dict], list[Run]]:
    """Every session record of a family, and each observed run of each animal as a corpus Run."""
    records, out = [], []
    for name, group in sorted(FAMILIES[family].items()):
        record, xy, mask = session(name, group, data_dir)
        records.append(record)
        for animal, first, last in runs(mask):
            frames = np.arange(first, last + 1)
            out.append(Run(source=family, recording=name, track=str(animal + 1), group=group, t=frames / record["fps"],
                           xy=xy[frames, animal], native_hz=record["fps"], frame_size=tuple(record["frame_size"]),
                           provenance={"frames": [first, last]}))  # fmt: skip
    return records, out


def body_length(data_dir: Path, name: str) -> float:
    """A session's animal length in frame units: idtracker.ai's estimate over the frame's longer side."""
    pixels = json.loads((data_dir / name / "validated_attributes.json").read_text(encoding="utf-8"))["body_length"]
    return pixels / max(json.loads((data_dir / name / "video_header.json").read_text(encoding="utf-8"))["size"])


def in_body_lengths(runs_: list[Run], data_dir: Path) -> tuple[list[Run], dict[str, float]]:
    """Each run's pixels scaled about the frame's centre so that its session's body length is
    REFERENCE_BODY frame units, and each session's factor. Cameras film the animals at different
    zooms, so in frame units one behaviour would have many speeds."""
    out, zooms = [], {}
    for run in runs_:
        if run.recording not in zooms:
            pixels = json.loads((data_dir / run.recording / "validated_attributes.json").read_text(encoding="utf-8"))["body_length"]
            zooms[run.recording] = REFERENCE_BODY * max(run.frame_size) / pixels
        centre = np.array(run.frame_size, dtype=np.float64) / 2
        out.append(dataclasses.replace(run, xy=centre + (run.xy - centre) * zooms[run.recording]))
    return out, zooms
