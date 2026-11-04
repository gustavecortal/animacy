"""The idtracker.ai recordings: download from the authors' public Google Drive folder, checks, and runs.

Each session folder holds `trajectories/with_gaps_csv/trajectories.csv` (the tracker's centroids, NaN where an
animal was not identified) and `trajectories/validated_csv/` (after human validation, complete). A position is
used only where the tracker measured it and validation kept it unchanged.
"""

import dataclasses
import functools
import html
import http.client
import io
import json
import re
import struct
import time
import urllib.error
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
    "inconsistent_frames.csv": "inconsistent_frames.csv",  # in some sessions only
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
EMPTY_FIELD = re.compile(r"(?m)(?<![^,\n])(?=,|\r?$)")  # where a CSV field is empty


def pinned() -> dict[str, dict[str, str]]:
    """The SHA-256 of every file the models were trained on, by session."""
    return json.loads(resources.files(__package__).joinpath("sessions.json").read_text(encoding="utf-8"))


def transient(error: Exception) -> bool:
    """Whether a failed request may succeed if repeated: a timeout, a dropped or refused connection, a cut
    read, or HTTP 408, 429 or 5xx. Other HTTP errors, such as 403, 404 or 416, will not change."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or error.code >= 500
    return isinstance(error, (OSError, http.client.HTTPException))


def retried(request, attempts: int = 8):
    """request() repeated with growing pauses (about 21 minutes in all) while it fails transiently."""
    for attempt in range(attempts):
        try:
            return request()
        except Exception as error:
            if not transient(error) or attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)


def check_quota(url: str, page: bytes) -> None:
    """Drive answers with a web page when a file's download quota is used up, which retrying soon does not fix."""
    if b"Quota exceeded" in page:
        raise RuntimeError(f"{url}: Google Drive's download quota for this file is used up. Run fetch again later: it keeps the files it has.")


def _get(url: str, start: int, end: int) -> tuple[bytes, int]:
    """Bytes start..end inclusive (fewer at the end of the file) and the file's size. Drive sometimes
    answers with a web page instead of the file, which is retried like a dropped connection."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})

    def once() -> tuple[bytes, int]:
        with urllib.request.urlopen(request, timeout=60) as response:
            if "text/html" in response.headers.get("Content-Type", ""):
                check_quota(url, response.read())
                raise ConnectionError(f"{url}: a web page instead of the file")
            if response.status == 206:
                return response.read(), int(response.headers["Content-Range"].rpartition("/")[2])
            if response.status == 200 and start == 0:  # Drive sends some small files whole
                data = response.read(end + 1)
                return data, int(response.headers.get("Content-Length") or len(data))
            raise ValueError(f"{url}: HTTP {response.status} to a range request")

    return retried(once)


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


@functools.cache  # a failed listing raises, so only complete ones are kept
def list_folder(folder_id: str) -> dict[str, str]:
    """Names and IDs of a public Drive folder's entries. A page without entries is retried, since none of
    the folders read here is empty."""
    url = FOLDER_URL.format(folder_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    def once() -> dict[str, str]:
        with urllib.request.urlopen(request, timeout=60) as response:
            page = response.read()
        check_quota(url, page)
        entries = {html.unescape(name): entry for entry, name in ENTRY.findall(page.decode())}
        if not entries:
            raise ConnectionError(f"{url}: a page without the folder's entries")
        return entries

    return retried(once)


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
    data = get_range(FILE_URL.format(find(VIDEOS, f"{session}.avi")), 0, 65535)
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise ValueError(f"{session}.avi is not an AVI file")
    i = data.find(b"avih")
    width, height = struct.unpack_from("<2I", data, i + 40)
    # A stream header is "strh", its size, then the stream type: the video stream's gives the rate.
    video = re.search(rb"strh....vids", data, re.S)
    if video is None:
        raise ValueError(f"{session}.avi has no video stream header in its first 64 KiB")
    i = video.start()
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
        if name == "video_header.json":
            data = json.dumps(video_header(session)).encode()
        else:
            data = get_range(FILE_URL.format(find(remote, SESSION_FILES[name])))
        if sha256(data) != expected[name]:
            raise ValueError(f"{session}/{name} differs from the file the models were trained on")
        write_atomic(path, data)


def load_positions(raw: bytes, path: Path) -> np.ndarray:
    """The bytes of a trajectories CSV (`x1, y1, x2, y2, ...` in pixels) as float64 [frames, animals, 2],
    NaN where nan or empty."""
    header, _, rows = raw.decode().partition("\n")
    header = header.rstrip("\r").split(",")
    if len(header) % 2 or header[:2] != ["x1", "y1"]:
        raise ValueError(f"{path}: unexpected header {header[:5]}")
    try:
        values = np.loadtxt(io.StringIO(rows), delimiter=",", ndmin=2)
    except ValueError:  # empty fields, which the pinned files do not have
        values = np.loadtxt(io.StringIO(EMPTY_FIELD.sub("nan", rows.rstrip())), delimiter=",", ndmin=2)
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
    files = {}
    for file, expected in pinned()[name].items():
        files[file] = (folder / file).read_bytes()
        if sha256(files[file]) != expected:
            raise ValueError(f"{folder / file} differs from the file the models were trained on: run fetch again")
    meta = json.loads(files["session.json"])
    header = json.loads(files["video_header.json"])
    with_gaps = load_positions(files["with_gaps.csv"], folder / "with_gaps.csv")
    validated = load_positions(files["validated.csv"], folder / "validated.csv")
    if with_gaps.shape != validated.shape or len(with_gaps) != header["frames"]:
        raise ValueError(f"{name}: exports {with_gaps.shape} and {validated.shape}, video {header['frames']} frames")
    if header["size"] != [meta["original_width"], meta["original_height"]] or meta["resolution_reduction"] != 1:
        raise ValueError(f"{name}: video {header['size']}, session {meta['original_width']}x{meta['original_height']}")
    excluded = [int(x) for x in files.get("inconsistent_frames.csv", b"").split()]
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


def model_runs(families: tuple[str, ...], body_lengths: bool, data_dir: Path) -> tuple[list[dict], list[Run], dict[str, float]]:
    """The session records and runs of a model's families, in body lengths if `body_lengths` (see
    in_body_lengths), and each rescaled session's zoom."""
    records, out, zooms = [], [], {}
    for family in families:
        r, f = family_runs(family, data_dir)
        if body_lengths:
            f, z = in_body_lengths(f, data_dir)
            zooms |= z
            # A tracking region is in pixels, wrong once positions are rescaled. It is dropped rather than
            # rescaled because the shipped scenes.json and seeds' arenas were built without it.
            r = [{**x, "roi": None} for x in r]
        records += r
        out += f
    return records, out, zooms


def body_pixels(data_dir: Path, name: str) -> tuple[float, int]:
    """A session's animal length in pixels, as idtracker.ai estimated it, and its frame's longer side."""
    pixels = json.loads((data_dir / name / "validated_attributes.json").read_text(encoding="utf-8"))["body_length"]
    return pixels, max(json.loads((data_dir / name / "video_header.json").read_text(encoding="utf-8"))["size"])


def body_length(data_dir: Path, corpus_dir: Path, name: str) -> float:
    """A session's animal length in the corpus's frame units: over the frame's longer side, times the
    session's zoom where the corpus is in body lengths (its zooms.json)."""
    pixels, side = body_pixels(data_dir, name)
    zooms = corpus_dir / "zooms.json"
    return pixels / side * (json.loads(zooms.read_text(encoding="utf-8")).get(name, 1.0) if zooms.exists() else 1.0)


def in_body_lengths(runs_: list[Run], data_dir: Path) -> tuple[list[Run], dict[str, float]]:
    """Each run's pixels scaled about the frame's centre so that its session's body length is
    REFERENCE_BODY frame units, and each session's factor. Cameras film the animals at different
    zooms, so in frame units one behaviour would have many speeds."""
    out, zooms = [], {}
    for run in runs_:
        if run.recording not in zooms:
            pixels, side = body_pixels(data_dir, run.recording)
            zooms[run.recording] = REFERENCE_BODY * side / pixels
        centre = np.array(run.frame_size, dtype=np.float64) / 2
        out.append(dataclasses.replace(run, xy=centre + (run.xy - centre) * zooms[run.recording]))
    return out, zooms
