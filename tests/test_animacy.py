"""Tests that need no download: the shipped models against the live page, the cache against the full pass, and
the whole pipeline on synthetic recordings."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from animacy import corpus, scenes, train
from animacy.export import export, load_export
from animacy.model import Config, Model, neighbour_inputs
from animacy.rollout import rollout

ROOT = Path(__file__).resolve().parent.parent
PAGE = json.loads((Path(__file__).parent / "page_outputs.json").read_text(encoding="utf-8"))


def synthetic_inputs(n: int, animals: int, scale: float):
    """The inputs page_outputs.json was computed on: tokens [n, 2], and every animal's seen position and last
    step [n, animals, 2], animal 3 unobserved from token 100 to 149."""
    i = np.arange(n, dtype=np.float64)[:, None]
    a = np.arange(animals, dtype=np.float64)[None, :]
    tokens = np.column_stack([0.9 * np.cos(0.05 * i) + 0.3 * np.sin(0.31 * i), 0.9 * np.sin(0.07 * i) - 0.2 * np.cos(0.23 * i)])
    seen = np.stack([scale * 5 * a * np.cos(0.02 * i + a), scale * 5 * a * np.sin(0.03 * i + 2 * a)], -1)
    step = np.stack([scale * 0.5 * np.cos(0.1 * i + a), scale * 0.5 * np.sin(0.1 * i + a)], -1)
    seen[100:150, 3] = step[100:150, 3] = np.nan
    return tokens, seen, step


def close_to_page(raw: np.ndarray, reference: list) -> float:
    """The worst error as a share of the page's parity tolerance, 1e-4 + 1e-3 |reference|."""
    reference = np.array(reference)
    return float((np.abs(raw - reference) / (1e-4 + 1e-3 * np.abs(reference))).max())


@pytest.mark.parametrize("name", ["fish", "flies", "both"])
def test_shipped_models_match_the_live_page(name):
    """Full pass in float64 and cached steps in float32 both give the browser model's raw output."""
    model = load_export(ROOT / "exports" / name)
    tokens, seen, step = synthetic_inputs(PAGE["tokens"], PAGE["animals"], model.scale)
    inputs = neighbour_inputs(torch.from_numpy(seen), torch.from_numpy(step), torch.zeros(len(tokens), dtype=torch.long), model.config.neighbours, model.scale)
    with torch.no_grad():
        full = model.double()(torch.from_numpy(tokens)[None], tuple(x[None] for x in inputs))[0].numpy()
        model.float()
        cache = model.cache(1)
        stepped = np.stack([model.step(torch.from_numpy(tokens[i : i + 1]).float(), cache, tuple(x[i : i + 1].float() if x.is_floating_point() else x[i : i + 1] for x in inputs))[0].numpy() for i in range(len(tokens))])
    indices = PAGE["indices"]
    assert close_to_page(full[indices], PAGE["raw"][name]) <= 1
    assert close_to_page(stepped[indices], PAGE["raw"][name]) <= 1


def test_cached_steps_equal_the_full_pass_through_cache_rollovers():
    torch.manual_seed(0)
    model = Model(Config(layers=2, heads=2, window=8, neighbours=3, head_windows=((8, 3), (5, 8)))).double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    tokens, seen, step = synthetic_inputs(40, 5, 0.01)
    inputs = neighbour_inputs(torch.from_numpy(seen), torch.from_numpy(step), torch.zeros(40, dtype=torch.long), 3, 0.01)
    with torch.no_grad():
        full = model(torch.from_numpy(tokens)[None], tuple(x[None] for x in inputs))[0]
        cache = model.cache(1)
        stepped = torch.stack([model.step(torch.from_numpy(tokens[i : i + 1]), cache, tuple(x[i : i + 1] for x in inputs))[0] for i in range(40)])
    torch.testing.assert_close(stepped, full, rtol=1e-10, atol=1e-10)


def synthetic_recordings(data_dir: Path) -> tuple[list[dict], list[corpus.Run]]:
    """Six sessions of three animals swimming loops for 70 s at 30 fps, each session its own group."""
    rng = np.random.default_rng(0)
    records, runs = [], []
    t = np.arange(2100) / 30.0
    for s in range(6):
        name = f"session_{s}"
        (data_dir / name).mkdir(parents=True)
        corpus.write_json(data_dir / name / "validated_attributes.json", {"body_length": 20.0})
        corpus.write_json(data_dir / name / "video_header.json", {"size": [1000, 1000]})
        records.append({"session": name, "group": name, "animals": 3, "frames": len(t), "fps": 30.0, "frame_size": [1000, 1000], "roi": None})
        for a in range(3):
            phase = rng.uniform(0, 2 * np.pi)
            xy = 500 + np.column_stack([200 * np.cos(0.3 * t + phase), 150 * np.sin(0.4 * t + phase)]) + rng.normal(0, 0.5, (len(t), 2)).cumsum(0)
            runs.append(corpus.Run("fish", name, str(a + 1), name, t, xy, 30.0, (1000, 1000)))
    return records, runs


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """A folder with the synthetic recordings' sessions/, corpus/ and scenes/, shared by the tests that read them."""
    root = tmp_path_factory.mktemp("synthetic")
    records, runs = synthetic_recordings(root / "sessions")
    corpus.build(runs, root / "corpus")
    scenes.build(runs, records, root / "corpus", root / "scenes")
    return root


@pytest.fixture
def short_windows(monkeypatch):
    monkeypatch.setattr(train, "LENGTH", 64)
    monkeypatch.setattr(train, "EVAL_EVERY", 10)


def test_train_export_and_rollout_on_synthetic_recordings(built, tmp_path, short_windows):
    train.train(built / "corpus", None, tmp_path / "solo", 10, 2e-3, 0.0, None, 4)
    train.train(built / "corpus", built / "scenes", tmp_path / "group", 10, 5e-4, 10.0, tmp_path / "solo" / "model.pt", 4)
    windows = ((64, 16, 16, 16), (16, 16, 16, 16), (16, 16, 16, 16), (16, 16, 16, 64))
    export(tmp_path / "group" / "model.pt", built / "corpus", built / "scenes", built / "sessions", tmp_path / "export", windows)
    manifest = json.loads((tmp_path / "export" / "model.json").read_text(encoding="utf-8"))
    assert manifest["config"]["head_windows"] == [list(row) for row in windows] and manifest["total_bytes"] == len((tmp_path / "export" / "weights.bin").read_bytes())
    fixtures = json.loads((tmp_path / "export" / "fixtures.json").read_text(encoding="utf-8"))
    model = load_export(tmp_path / "export").double()
    inputs = neighbour_inputs(*(torch.tensor(np.array(fixtures[k], dtype=float)) for k in ("seen", "seen_step")), torch.full((len(fixtures["tokens"]),), fixtures["animal"]), 16, manifest["scale"])
    with torch.no_grad():
        raw = model(torch.tensor(fixtures["tokens"], dtype=torch.float64)[None], tuple(x[None] for x in inputs))[0].numpy()
    np.testing.assert_allclose(raw[fixtures["indices"]], fixtures["raw"], rtol=1e-9, atol=1e-9)
    summary = rollout(tmp_path / "export", tmp_path / "school.gif", 1.0, seconds=1.0, recorded=(built / "corpus", built / "scenes", built / "sessions"))
    assert summary["animals"] == 3 and (tmp_path / "school.gif").stat().st_size > 0


@pytest.mark.parametrize("name", ["fish", "flies", "both"])
def test_shipped_models_roll_out_without_the_recordings(name, tmp_path):
    summary = rollout(ROOT / "exports" / name, tmp_path / "school.gif", 1.0, seconds=1.0)
    assert summary["animals"] == 10 and (tmp_path / "school.gif").stat().st_size > 0


def test_train_reruns_only_finished_stages_with_the_recipe_settings(built, tmp_path, short_windows, monkeypatch, capsys):
    """A short --steps check is not taken for a finished stage by a later full run."""
    from animacy.configs import Stage

    monkeypatch.setattr(train, "STAGES", (Stage("solo", 6, 2e-3, 0.0, False), Stage("group", 6, 5e-4, 10.0, True)))
    stages = lambda stage=None, steps=None: train.train_stages(built / "corpus", built / "scenes", tmp_path, stage, steps, 4)
    stages(steps=2)
    with pytest.raises(ValueError, match="remove it or pass --out"):
        stages()
    for stage in ("solo", "group"):
        for f in (tmp_path / stage).iterdir():
            f.unlink()
    stages()
    assert torch.load(tmp_path / "group" / "model.pt", weights_only=False)["step"] == 6
    capsys.readouterr()
    stages()
    assert capsys.readouterr().out.count("skipped") == 2
    (tmp_path / "group" / "model.pt").unlink()  # resumed from a last.pt at the final step, it still has a validation loss
    stages("group")
    assert np.isfinite(torch.load(tmp_path / "group" / "model.pt", weights_only=False)["val"])
