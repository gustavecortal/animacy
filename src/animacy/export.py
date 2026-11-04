"""Export a checkpoint for the browser: model.json, weights.bin, fixtures.json and seeds.json.

weights.bin holds every tensor of the state dict as little-endian float16, row-major, in manifest order.
model.json names each tensor's shape and byte range and spells out every operation, so the browser needs no
Python default. fixtures.json holds one validation animal's tokens, what it saw of its neighbours, and the full
model's raw head output for them, computed in float64, for checking the browser's incremental implementation.
seeds.json holds the page's recorded starts: three 2 s paths and 2 s of every animal of each validation scene.
"""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import corpus, data, scenes
from .corpus import CONVENTION, RATE_HZ, sha256, write_json
from .model import RHO_LIMIT, SEED, SIGMA_FLOOR, Config, Model, load_checkpoint, neighbour_inputs

VERSION = 4  # the page's model format
SCHOOLS_PER_SCENE = 2
FIXTURE_TOKENS = 1100  # beyond four cache rollovers at a window of 256
FIXTURE_INDICES = [0, 1, 2, 100, 254, 255, 256, 257, 300, 511, 512, 513, 767, 768, 1023, 1024, 1099]
SEED_S, CONTINUE_S = 2, 20  # a drawn seed and the continuation it must have in the data
SEED_NAMES = {"Swimming": "Typical", "Turning": "Turning", "Resting": "Near stillness"}

DEFINITIONS = {
    "token": "displacement between consecutive 30 Hz positions divided by scale",
    "linear": "y = W x (+ bias where the manifest lists one), W row-major [out, in]",
    "rms_norm": "x / sqrt(mean(x^2) + norm_eps) * weight",
    "embed": "h = embed.weight token, plus neighbours(token, ...) when config.neighbours > 0",
    "neighbour_inputs": "for each other animal with a finite position and step: relative = (its position - own) / scale, "
    "displacement = its last step / scale. Keep the config.neighbours nearest by |relative|, nearest first; the rest "
    "are not visible",
    "neighbours": "per visible neighbour, f = [token_x, token_y, relative / d, log d, displacement], d = max(|relative|, "
    "0.1); e = gelu(social.encode.2 gelu(social.encode.0 f)); weights = softmax over visible neighbours of "
    "social.score e; result = social.out (sum of weights e), zero when none is visible",
    "layer": "a = rms_norm(h, attn_norm); q, k, v = rows [0, width), [width, 2 width), [2 width, 3 width) of "
    "qkv.weight a; head i uses entries [i head_dim, (i + 1) head_dim) of each; q and k of each head go through "
    "rms_norm with q_norm and k_norm, then RoPE; h += out.weight attention(q, k, v); "
    "h += down.weight gelu(up.weight rms_norm(h, ff_norm))",
    "rope": "pairs (x[2m], x[2m + 1]) of a head vector rotated by angle position * rope_base^(-2m / head_dim), "
    "as (x0 cos - x1 sin, x0 sin + x1 cos). Scores depend only on query position minus key position",
    "attention": "per head, softmax over the current token and at most w - 1 before it of q.k / sqrt(head_dim), "
    "weights applied to v, heads concatenated, w being config.head_windows[layer][head], or config.window where it is null",
    "gelu": "exact: x / 2 (1 + erf(x / sqrt(2)))",
    "head": "raw = head.weight rms_norm(h, norm): components x [logit, mean_x, mean_y, raw_sigma_x, raw_sigma_y, raw_rho]",
    "mixture": "weights softmax(logit / temperature), sigma = softplus(raw_sigma) + sigma_floor, "
    "rho = rho_limit tanh(raw_rho), all in token units",
}


def _nulls(a: np.ndarray) -> list:
    """`a` as nested lists of floats, its nonfinite values as None (JSON null)."""
    return np.where(np.isfinite(a), a.astype(np.float64), None).tolist()


def page_config(c: Config) -> dict:
    """The config as model.json names it. The page's format also has a wall input, always off here."""
    out = asdict(c)
    head_windows = out.pop("head_windows")
    return out | {"walls": False, "head_windows": head_windows}


def fixtures(model: Model, scale: float, scenes_dir: Path) -> dict:
    """The first animal of the validation scene with fewest animals observed for FIXTURE_TOKENS + 1 rows: its
    tokens, every animal's `seen` and `seen_step` at each token's instant (null where unobserved), its index,
    and raw head output at FIXTURE_INDICES with neighbours and without."""
    for _, session in sorted((e["animals"], s) for s, e in scenes.sessions(scenes_dir, "validation").items()):
        positions, seen, seen_step = scenes.load(scenes_dir, session, "positions", "seen", "seen_step")
        long = [(j, first) for j, first, last in data.runs(np.isfinite(positions[..., 0])) if last - first > FIXTURE_TOKENS]
        if long:
            break
    else:
        raise ValueError(f"no validation animal is observed for more than {FIXTURE_TOKENS} rows")
    me, first = long[0]
    tokens = np.diff(positions[first : first + FIXTURE_TOKENS + 1, me].astype(np.float64), axis=0) / scale
    now = np.arange(first + 1, first + FIXTURE_TOKENS + 1)  # token i is the step to row now[i]
    double = lambda a: torch.from_numpy(a[now].astype(np.float64))
    inputs = neighbour_inputs(double(seen), double(seen_step), torch.full((FIXTURE_TOKENS,), me), model.config.neighbours, scale)
    model.double()
    with torch.no_grad():
        t = torch.from_numpy(tokens)[None]
        raw = model(t, tuple(x[None] for x in inputs))[0].numpy()
        alone = model(t)[0].numpy()
    model.float()
    return {
        "session": session, "animal": int(me), "first_row": int(first),
        "tokens": tokens.tolist(), "seen": _nulls(seen[now]), "seen_step": _nulls(seen_step[now]),
        "indices": FIXTURE_INDICES, "raw": raw[FIXTURE_INDICES].tolist(), "raw_alone": alone[FIXTURE_INDICES].tolist(),
    }  # fmt: skip


def describe(xy: np.ndarray) -> tuple[float, float]:
    """Median speed and absolute turning per second, from 0.25 s displacements."""
    lag = RATE_HZ // 4
    d = xy[lag:] - xy[:-lag]
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.median(np.linalg.norm(d, axis=1)) * RATE_HZ / lag), float(np.abs(np.diff(heading)).sum() / (len(xy) / RATE_HZ))


def pick_seeds(positions: np.ndarray, offsets: np.ndarray) -> dict[str, np.ndarray]:
    """Validation windows of 2 s plus 20 s, typical, turning and near stillness, each from its own sequence,
    among 2,000 drawn at random."""
    n = (SEED_S + CONTINUE_S) * RATE_HZ + 1
    rng = np.random.default_rng(SEED)
    long = [k for k in range(len(offsets) - 1) if offsets[k + 1] - offsets[k] >= n]
    windows = []
    for k in rng.permutation(long)[:2000]:
        start = rng.integers(offsets[k], offsets[k + 1] - n + 1)
        windows.append((k, positions[start : start + n]))
    stats = [describe(w[: SEED_S * RATE_HZ + 1]) for _, w in windows]
    speed = np.array([s for s, _ in stats])
    moving = [i for i, s in enumerate(speed) if s >= np.median(speed)]
    order = {
        "Typical": sorted(range(len(windows)), key=lambda i: abs(speed[i] - np.median(speed))),
        "Near stillness": sorted(range(len(windows)), key=lambda i: speed[i]),
        "Turning": sorted(moving, key=lambda i: -stats[i][1]),
    }
    chosen, used = {}, set()
    for label, indices in order.items():
        i = next(i for i in indices if windows[i][0] not in used)
        used.add(windows[i][0])
        chosen[label] = windows[i][1]
    return chosen


def seeds(corpus_dir: Path, scenes_dir: Path, data_dir: Path) -> dict:
    """The page's recorded starts, in corpus units. `fish`, by name: three 2 s validation paths, each centred
    on its own bounding box. `schools`: 2 s of every animal of each validation scene at SCHOOLS_PER_SCENE
    starts, centred on the scene's arena, with what the tracker had observed by each row (`seen` and
    `seen_step`, null where nothing was). `body` is an animal's length in corpus units, as idtracker.ai
    estimated it, per school and, for a drawn path, the median over the training sessions. `pace` is the
    90th percentile of the training animals' speeds, corpus units per second, which sizes a drawing's square."""
    positions, offsets, _ = corpus.load(corpus_dir, "validation")
    picked = pick_seeds(positions, offsets)
    chosen = {}
    for name, label in SEED_NAMES.items():
        seed = picked[label][: SEED_S * RATE_HZ + 1].astype(np.float64)
        seed -= (seed.min(axis=0) + seed.max(axis=0)) / 2
        chosen[name] = np.round(seed, 6).tolist()
    length = lambda name: data.body_length(data_dir, corpus_dir, name)
    mine = [length(r) for r in set(corpus.sequence_field(corpus_dir, "train", "recording"))]
    train, train_offsets, _ = corpus.load(corpus_dir, "train")
    moved = np.linalg.norm(np.diff(train.astype(np.float64), axis=0), axis=-1) * RATE_HZ
    within = np.ones(len(moved), bool)
    within[train_offsets[1:-1] - 1] = False  # no step joins two sequences
    out = {"fish": chosen, "body": round(float(np.median(mine)), 5), "pace": round(float(np.percentile(moved[within], 90)), 5), "schools": []}
    for session, entry in scenes.sessions(scenes_dir, "validation").items():
        positions, seen, seen_step = scenes.load(scenes_dir, session, "positions", "seen", "seen_step")
        lo, hi = scenes.arena(entry, positions)
        centre = (lo + hi) / 2
        rows = lambda a, start, shift=0: np.round(a[start - scenes.SEED_ROWS : start + 1].astype(np.float64) - shift, 6).transpose(1, 0, 2)
        for start in scenes.starts(positions, SCHOOLS_PER_SCENE):
            out["schools"].append({"session": session, "start_s": start / RATE_HZ, "arena": np.round(hi - lo, 6).tolist(), "body": round(length(session), 5),
                                   "fish": rows(positions, start, centre).tolist(), "seen": _nulls(rows(seen, start, centre)),
                                   "seen_step": _nulls(rows(seen_step, start))})  # fmt: skip
    return out


def export(checkpoint: Path, corpus_dir: Path, scenes_dir: Path, data_dir: Path, out: Path, head_windows) -> dict:
    """Write the four files to out/, each head attending to its window of `head_windows`, the fixtures included,
    so the browser's parity check covers the model it runs."""
    model, ckpt = load_checkpoint(checkpoint)
    if not math.isfinite(ckpt["val"]):
        raise ValueError(f"{checkpoint} has no validation loss")
    windowed = Model(replace(model.config, head_windows=tuple(map(tuple, head_windows))))
    windowed.load_state_dict(model.state_dict())
    model, scale = windowed.eval(), ckpt["scale"]
    if not model.config.neighbours:
        raise ValueError(f"{checkpoint} sees no neighbours: export a group stage")
    # The page downloads float16 weights, and the fixtures come from the same rounded model.
    with torch.no_grad():
        for t in model.state_dict().values():
            t.copy_(t.half().float())
    tensors, chunks, offset = [], [], 0
    for name, t in model.state_dict().items():
        chunk = t.half().numpy().astype("<f2").tobytes()
        tensors.append({"name": name, "shape": list(t.shape), "offset": offset, "bytes": len(chunk)})
        chunks.append(chunk)
        offset += len(chunk)
    weights = b"".join(chunks)
    corpus_meta = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "trajectory-model",
        "version": VERSION,
        "config": page_config(model.config),
        "rate_hz": RATE_HZ,
        "scale": scale,
        "convention": CONVENTION,
        "sigma_floor": SIGMA_FLOOR,
        "rho_limit": RHO_LIMIT,
        "dtype": "float16-le",
        "definitions": DEFINITIONS,
        "tensors": tensors,
        "total_bytes": len(weights),
        "weights_sha256": sha256(weights),
        "corpus": {"arrays": {p: a["sha256"] for p, a in corpus_meta["arrays"].items()}, "split_manifest_sha256": corpus_meta["split_manifest_sha256"]},
        "checkpoint": {"step": ckpt["step"], "validation_nll": ckpt["val"]},
    }
    checked = fixtures(model, scale, scenes_dir)
    recorded = seeds(corpus_dir, scenes_dir, data_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(weights)
    write_json(out / "model.json", manifest)
    if not all(torch.equal(t, s) for t, s in zip(load_export(out).state_dict().values(), model.state_dict().values())):
        raise ValueError("the exported weights do not reload as the model's")
    write_json(out / "fixtures.json", checked, indent=None)
    write_json(out / "seeds.json", recorded, indent=None)
    return {k: manifest[k] for k in ("total_bytes", "weights_sha256", "checkpoint")}


def load_export(directory: Path) -> Model:
    """The model of an export's model.json and weights.bin, in float32, as the page runs it."""
    manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    weights = (directory / "weights.bin").read_bytes()
    if manifest["version"] != VERSION or sha256(weights) != manifest["weights_sha256"]:
        raise ValueError(f"{directory}: unknown version or weights that do not match model.json")
    config = {k: v for k, v in manifest["config"].items() if k != "walls"}
    model = Model(Config(**{**config, "head_windows": tuple(map(tuple, config["head_windows"]))}))
    state = {t["name"]: torch.from_numpy(np.frombuffer(weights, "<f2", t["bytes"] // 2, t["offset"]).astype(np.float32).reshape(t["shape"])) for t in manifest["tensors"]}
    model.load_state_dict(state)
    model.scale = manifest["scale"]
    return model.eval()
