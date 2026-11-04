"""A held-out school continued by an exported model, drawn as an animated GIF.

Every animal of a validation scene is seeded with 2 s of its recorded motion, then all of them step together,
each seeing the others' generated positions as it saw recorded ones in training. Steps are folded back into
the scene's arena, as the page folds animals into its square.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import data, scenes
from .corpus import RATE_HZ
from .export import SCHOOLS_PER_SCENE, load_export
from .model import fold, neighbour_inputs, sample

SIZE = 480  # pixels
FPS = 15  # frames per second of the GIF: every second model step
TRAIL_S = 1.0


@torch.no_grad()
def simulate(model, positions, seen, seen_step, start: int, seconds: float, box, temperature: float, seed: int) -> np.ndarray:
    """Positions [rows, animals, 2]: the 2 s seed of every animal before row `start`, then `seconds` of
    model steps, each animal seeing its nearest neighbours' latest positions and steps."""
    k, scale = model.config.neighbours, model.scale
    animals = positions.shape[1]
    me = torch.arange(animals)
    lo, hi = (torch.from_numpy(np.asarray(b, dtype=np.float64)) for b in box)
    generator = torch.Generator().manual_seed(seed)
    cache = model.cache(animals)

    def step(token, here, moved):
        relative, displacement, visible = neighbour_inputs(here.float()[None].expand(animals, -1, -1), moved.float()[None].expand(animals, -1, -1), me, k, scale)
        return model.step((token / scale).float(), cache, (relative, displacement, visible))

    rows = np.arange(start - scenes.SEED_ROWS, start + 1)
    track = torch.from_numpy(positions[rows]).double()
    for i in range(scenes.SEED_ROWS):
        raw = step(track[i + 1] - track[i], torch.from_numpy(seen[rows[i + 1]]), torch.from_numpy(seen_step[rows[i + 1]]))
    end, out = track[-1], list(track)
    for t in range(round(seconds * RATE_HZ)):
        proposal = fold(end + sample(raw, generator, temperature) * scale, lo, hi)
        moved, end = proposal - end, proposal
        out.append(end)
        raw = step(moved, end, moved)
    return torch.stack(out).numpy()


def draw(path: np.ndarray, box, body: float, dest: Path) -> None:
    """Each animal as a dot with a fading trail, on a plain background, one frame per 1 / FPS seconds."""
    lo, hi = (np.asarray(b) for b in box)
    side = (hi - lo).max() * 1.04
    centre = (lo + hi) / 2
    to_px = lambda p: np.column_stack([(p[..., 0] - centre[0]) / side + 0.5, 0.5 - (p[..., 1] - centre[1]) / side]) * SIZE
    radius = max(2.0, body / side * SIZE / 2)
    every = RATE_HZ // FPS
    trail = round(TRAIL_S * RATE_HZ)
    frames = []
    for row in range(scenes.SEED_ROWS, len(path), every):
        image = Image.new("L", (SIZE, SIZE), 255)
        pen = ImageDraw.Draw(image)
        for animal in range(path.shape[1]):
            points = to_px(path[max(0, row - trail) : row + 1, animal])
            for i in range(1, len(points)):
                shade = round(235 - 135 * i / len(points))
                pen.line([tuple(points[i - 1]), tuple(points[i])], fill=shade, width=max(1, round(radius / 2)))
            x, y = points[-1]
            pen.ellipse([x - radius, y - radius, x + radius, y + radius], fill=30)
        frames.append(image)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(dest, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, optimize=True)


def rollout(export_dir: Path, corpus_dir: Path, scenes_dir: Path, data_dir: Path, dest: Path, temperature: float, seconds: float = 20.0, seed: int = 0) -> dict:
    """Continue the school at the first recorded start of the first validation scene and draw it to `dest`."""
    model = load_export(export_dir)
    session, entry = next(iter(scenes.sessions(scenes_dir, "validation").items()))
    positions, seen, seen_step = scenes.load(scenes_dir, session, "positions", "seen", "seen_step")
    start = scenes.starts(positions, SCHOOLS_PER_SCENE)[0]
    box = scenes.arena(entry, positions)
    path = simulate(model, positions, seen, seen_step, start, seconds, box, temperature, seed)
    zooms = corpus_dir / "zooms.json"
    body = data.body_length(data_dir, session) * (json.loads(zooms.read_text())[session] if zooms.exists() else 1.0)
    draw(path, box, body, dest)
    return {"session": session, "start_s": start / RATE_HZ, "animals": positions.shape[1], "seconds": seconds, "gif": str(dest)}
