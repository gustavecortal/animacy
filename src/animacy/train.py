"""Training: uniform window starts, every target scored, random rotations and mirrors, resumable.

A window is up to 256 consecutive tokens of one sequence, each predicting the next. Windows start uniformly over
all tokens that have a next one, so early targets see a short history, as a short seed does. Padding beyond a
sequence's end is excluded from the loss and, being after every real token, from their attention.

Each window is rotated by a random angle and mirrored with probability 1/2: the model sees displacements only,
never positions, so a turned or mirrored path is equally valid.
"""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import corpus, data, scenes
from .corpus import write_json
from .model import SEED, Config, Model, load_checkpoint, mixture_mean, neighbour_inputs, nll

NEIGHBOURS = 16  # most neighbours an animal sees
KEEP = (0, 1, 2, 4, 8, 16)  # nearest neighbours a training batch shows, drawn uniformly per batch
LENGTH = 256  # tokens per window, and each layer's attention window
EVAL_EVERY = 250  # updates between validation and checkpoints
LOG_EVERY = 25


class Windows:
    """Token windows over one partition of the corpus, held on `device`."""

    def __init__(self, corpus_dir: Path, partition: str, device):
        positions, offsets, self.scale = corpus.load(corpus_dir, partition)
        self.hold([np.diff(positions[a:b], axis=0) / self.scale for a, b in zip(offsets[:-1], offsets[1:])], device)

    def hold(self, sequences: list[np.ndarray], device):
        """Concatenate each sequence's tokens and note where each one's sequence ends."""
        lengths = np.array([len(s) for s in sequences])
        end = np.repeat(np.cumsum(lengths), lengths)
        self.starts = np.flatnonzero(np.arange(len(end)) + 1 < end)
        self.tokens = torch.from_numpy(np.concatenate(sequences).astype(np.float32)).to(device)
        self.end = torch.from_numpy(end).to(device)

    def index(self, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Token indices [b, LENGTH] of each window, padding repeating its start, and the mask of real targets."""
        s = torch.from_numpy(starts).to(self.tokens.device)
        idx = s[:, None] + torch.arange(LENGTH, device=s.device)
        valid = idx + 1 < self.end[s][:, None]
        return torch.where(valid, idx, s[:, None]), valid

    def gather(self, idx, valid, rotation):
        inputs = self.tokens[idx] * valid[..., None]
        targets = self.tokens[torch.where(valid, idx + 1, idx)] * valid[..., None]
        if rotation is not None:
            r = torch.from_numpy(rotation).to(idx.device)
            inputs, targets = inputs @ r.transpose(1, 2), targets @ r.transpose(1, 2)
        return inputs, targets, valid

    def batch(self, starts: np.ndarray, rotation: np.ndarray | None = None, keep: int = 0):
        """Inputs and targets [b, LENGTH, 2], the mask of real targets, and no neighbours."""
        return *self.gather(*self.index(starts), rotation), None


class SceneWindows(Windows):
    """Windows of one animal's tokens from the scenes, each with what it sees of its neighbours.

    A sequence is a run of rows where the animal's filtered track exists. Token i is its step to row r,
    and the neighbour input beside it is what was observed by row r's instant, from the scenes' unfiltered
    `seen` and `seen_step`."""

    def __init__(self, scenes_dir: Path, partition: str, device):
        self.scale = json.loads((scenes_dir / "scenes.json").read_text(encoding="utf-8"))["scale"]
        arrays = [scenes.load(scenes_dir, s, "positions", "seen", "seen_step") for s in scenes.sessions(scenes_dir, partition)]
        width = max(a.shape[1] for a, _, _ in arrays)
        rows = sum(len(a) for a, _, _ in arrays)
        seen_all, step_all = (np.full((rows, width, 2), np.nan, dtype=np.float32) for _ in range(2))
        tokens, now, animal = [], [], []
        base = 0
        for a, seen, step in arrays:
            seen_all[base : base + len(a), : a.shape[1]] = seen
            step_all[base : base + len(a), : a.shape[1]] = step
            for j, first, last in data.runs(np.isfinite(a[..., 0])):
                if last - first >= 2:  # three rows, as the corpus requires
                    tokens.append(np.diff(a[first : last + 1, j], axis=0) / self.scale)
                    now.append(base + np.arange(first + 1, last + 1))
                    animal.append(np.full(last - first, j))
            base += len(a)
        self.hold(tokens, device)
        self.now = torch.from_numpy(np.concatenate(now)).to(device)
        self.animal = torch.from_numpy(np.concatenate(animal)).to(device)
        self.seen = torch.from_numpy(seen_all).to(device)
        self.seen_step = torch.from_numpy(step_all).to(device)

    def batch(self, starts: np.ndarray, rotation: np.ndarray | None = None, keep: int = NEIGHBOURS):
        """As Windows.batch, with (relative, displacement, visible) of the `keep` nearest neighbours of each
        step, or None for none."""
        idx, valid = self.index(starts)
        inputs, targets, valid = self.gather(idx, valid, rotation)
        if not keep:
            return inputs, targets, valid, None
        now = self.now[idx]
        relative, displacement, visible = neighbour_inputs(self.seen[now], self.seen_step[now], self.animal[idx], NEIGHBOURS, self.scale)
        visible = (visible & valid[..., None])[:, :, :keep]
        relative, displacement = relative[:, :, :keep] * visible[..., None], displacement[:, :, :keep] * visible[..., None]
        if rotation is not None:
            r = torch.from_numpy(rotation).to(idx.device)[:, None]
            relative, displacement = relative @ r.transpose(-1, -2), displacement @ r.transpose(-1, -2)
        return inputs, targets, valid, (relative, displacement, visible)


def random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random rotations, each followed by a mirror of x with probability 1/2, as float32 [n, 2, 2]."""
    a = rng.uniform(0, 2 * np.pi, n)
    mirror = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    r = np.stack([np.stack([np.cos(a), -np.sin(a)], -1), np.stack([np.sin(a), np.cos(a)], -1)], -2)
    r[:, :, 0] *= mirror[:, None]
    return r.astype(np.float32)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward(model: Model, inputs: torch.Tensor, neighbours) -> torch.Tensor:
    """Raw head output in float32, with the trunk in bfloat16 on a CUDA or Intel GPU."""
    with torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=inputs.device.type in ("cuda", "xpu")):
        return model(inputs, neighbours).float()


def mean_loss(raw: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Squared error of the mixture's mean step per real target. Likelihood fits the density's bulk and can
    leave its mean off by a drift that is small at each step but adds up over seconds, such as the turn back
    toward the group. This term's minimum is the true conditional mean."""
    return (((targets - mixture_mean(raw)) ** 2).sum(-1) * valid).sum() / valid.sum()


@torch.no_grad()
def evaluate(model: Model, windows: Windows, starts: np.ndarray, batch: int) -> float:
    """Validation NLL per target, with every neighbour shown."""
    model.eval()
    total = count = 0.0
    for i in range(0, len(starts), batch):
        inputs, targets, valid, neighbours = windows.batch(starts[i : i + batch])
        total += (nll(forward(model, inputs, neighbours), targets) * valid).sum().item()
        count += valid.sum().item()
    model.train()
    return total / count


def train(corpus_dir: Path, scenes_dir: Path | None, out: Path, steps: int, lr: float, mean_weight: float, init: Path | None,
          batch: int, final_lr_fraction: float = 0.1, warmup_fraction: float = 0.1, weight_decay: float = 0.1) -> dict:  # fmt: skip
    """Train with linear warmup over `warmup_fraction` of the steps then cosine decay to `final_lr_fraction`,
    and write out/model.pt. With `scenes_dir`, the model sees its neighbours and starts from `init`, and each
    batch shows only one of KEEP nearest. Resumes from out/last.pt if present."""
    dev = device()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    social = scenes_dir is not None
    source = (lambda p: SceneWindows(scenes_dir, p, dev)) if social else (lambda p: Windows(corpus_dir, p, dev))
    train_w, val_w = source("train"), source("validation")
    rng.choice(train_w.starts, batch, replace=False)  # a draw the shipped runs made, kept so their batches are reproduced
    val_starts = np.random.default_rng(SEED + 1).choice(val_w.starts, min(1024, len(val_w.starts)), replace=False)
    model = (load_checkpoint(init, NEIGHBOURS if social else None)[0] if init else Model(Config(window=LENGTH))).to(dev)
    matrices = [p for p in model.parameters() if p.ndim >= 2]  # weight decay on matrices only, not on norm scales
    others = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": matrices, "weight_decay": weight_decay}, {"params": others, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    warmup = max(1, round(warmup_fraction * steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return final_lr_fraction + (1 - final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    run = {"config": asdict(model.config), "scale": train_w.scale, "init": str(init) if init else None, "steps": steps,
           "lr": lr, "mean_weight": mean_weight, "batch": batch, "device": str(dev), "torch": torch.__version__}  # fmt: skip
    step, last = 0, out / "last.pt"
    if last.exists():
        ckpt = torch.load(last, weights_only=False, map_location=dev)
        if {k: v for k, v in ckpt["run"].items() if k not in ("device", "torch")} != {k: v for k, v in run.items() if k not in ("device", "torch")}:
            raise ValueError(f"{last} was trained with other settings: remove {out} to start again")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        rng.bit_generator.state = ckpt["rng"]
        step = ckpt["step"]
    log = (out / "log.jsonl").open("a", encoding="utf-8", newline="\n")
    started = time.time()
    sums, since, skipped = torch.zeros(2, device=dev), 0, 0
    val = math.nan
    while step < steps:
        starts = rng.choice(train_w.starts, batch)
        keep = int(rng.choice(KEEP)) if social else 0
        inputs, targets, valid, neighbours = train_w.batch(starts, random_rotations(rng, batch), keep)
        raw = forward(model, inputs, neighbours)
        loss = (nll(raw, targets) * valid).sum() / valid.sum()
        squared = mean_loss(raw, targets, valid) if mean_weight else torch.zeros((), device=dev)
        # Divided by its own value, held constant, the squared error pulls on the mean about as much as the
        # likelihood does at mean_weight 1.
        opt.zero_grad(set_to_none=True)
        (loss + mean_weight * squared / squared.detach().clamp_min(1e-12)).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # A rare overflow in the bfloat16 trunk gives a nonfinite gradient: that update is dropped.
        if torch.isfinite(norm) & torch.isfinite(loss) & torch.isfinite(squared):
            opt.step()
            sums += torch.stack([loss.detach(), squared.detach()])
            since += 1
        else:
            skipped += 1
        sched.step()
        step += 1
        evaluating = step % EVAL_EVERY == 0 or step == steps
        if step % LOG_EVERY and not evaluating:
            continue
        if skipped >= LOG_EVERY:
            raise FloatingPointError(f"nonfinite loss or gradient before step {step}")
        entry = {"step": step, "minutes": round((time.time() - started) / 60, 2), "lr": sched.get_last_lr()[0]}
        if since:
            entry |= {"train_nll": sums[0].item() / since, **({"mean_loss": sums[1].item() / since} if mean_weight else {})}
        if evaluating:
            entry["val_nll"] = val = evaluate(model, val_w, val_starts, batch)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "step": step, "run": run}, out / "last.partial")  # fmt: skip
            (out / "last.partial").replace(last)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in entry.items()}), flush=True)
        sums.zero_()
        since, skipped = 0, 0
    if math.isnan(val):  # resumed from a last.pt saved at the final step
        val = evaluate(model, val_w, val_starts, batch)
    # The keys of the site's checkpoints, so export reads either.
    torch.save({"model": model.state_dict(), "config": run["config"], "scale": train_w.scale, "run": run, "step": step, "val": val}, out / "model.pt")
    write_json(out / "summary.json", {**run, "validation_nll": val, "minutes": (time.time() - started) / 60})
    return run
