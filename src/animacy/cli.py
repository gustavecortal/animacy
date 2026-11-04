"""animacy: fetch the recordings, build the corpus, train, export for the browser, and draw a rollout.

    animacy rollout fish      out/fish.gif: the exported model moving a school of 10 fish
    animacy fetch fish        download the fish sessions (flies, both: the fly sessions or all)
    animacy build fish        the corpus and scenes in data/fish/
    animacy train fish        the three training stages in runs/fish/
    animacy export fish       exports/fish/ from the last stage
"""

import argparse
import json
import sys
from pathlib import Path

from .configs import BATCH, MODELS, STAGES

DATA = Path("data")
SESSIONS = DATA / "sessions"


def paths(name: str) -> tuple[Path, Path]:
    return DATA / name / "corpus", DATA / name / "scenes"


def fetch(name: str) -> int:
    from . import data

    failed = []
    for family in MODELS[name].families:
        for session in sorted(data.FAMILIES[family]):
            print(session, flush=True)
            try:
                data.fetch(session, SESSIONS)
            except Exception as error:  # a used-up quota or a missing file: the other sessions can still come
                print(f"{session}: {type(error).__name__}: {error}", file=sys.stderr)
                failed.append(session)
    if failed:
        print(f"not fetched: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


def build(name: str) -> None:
    from . import corpus, data, scenes

    corpus_dir, scenes_dir = paths(name)
    records, runs, zooms = data.model_runs(MODELS[name].families, MODELS[name].body_lengths, SESSIONS)
    summary = corpus.build(runs, corpus_dir, zooms)
    built = scenes.build(runs, records, corpus_dir, scenes_dir)
    print(json.dumps({"scale": summary["scale"], "corpus": {p: a["sequences"] for p, a in summary["arrays"].items()}, "scenes": built["partitions"]}, indent=1))


def export(name: str, checkpoint: Path | None, out: Path | None) -> None:
    import torch

    from .export import export

    checkpoint = checkpoint or Path("runs") / name / STAGES[-1].name / "model.pt"
    step = torch.load(checkpoint, map_location="cpu", weights_only=False)["step"]
    if out is None and step != STAGES[-1].steps:
        raise SystemExit(f"{checkpoint} was trained for {step} steps, not {STAGES[-1].steps}: pass --out to export it outside exports/")
    print(json.dumps(export(checkpoint, *paths(name), SESSIONS, out or Path("exports") / name, MODELS[name].head_windows), indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animacy", description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog="\n".join(__doc__.splitlines()[2:]))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("rollout", "fetch", "build", "train", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("model", choices=MODELS)
        if command == "train":
            sub.add_argument("--stage", choices=[s.name for s in STAGES], help="run this stage only, from the previous one's model.pt")
            sub.add_argument("--steps", type=int, help="fewer steps than the recipe's, for a short check")
            sub.add_argument("--batch", type=int, default=BATCH, help=f"windows per step (default {BATCH})")
            sub.add_argument("--out", type=Path, help="runs/<model> by default")
        if command == "export":
            sub.add_argument("--checkpoint", type=Path, help=f"runs/<model>/{STAGES[-1].name}/model.pt by default")
            sub.add_argument("--out", type=Path, help="exports/<model> by default")
        if command == "rollout":
            sub.add_argument("--recorded", action="store_true", help="start from a recorded school of a validation scene, which needs fetch and build")
            sub.add_argument("--export", type=Path, help="exports/<model> by default")
            sub.add_argument("--seconds", type=float, default=20.0)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--out", type=Path, help="out/<model>.gif by default")
    args = parser.parse_args(argv)
    name = args.model
    if args.command == "fetch":
        return fetch(name)
    if args.command == "build":
        build(name)
    elif args.command == "train":
        from .train import train_stages

        train_stages(*paths(name), args.out or Path("runs") / name, args.stage, args.steps, args.batch)
    elif args.command == "export":
        export(name, args.checkpoint, args.out)
    elif args.command == "rollout":
        from .rollout import rollout

        recorded = (*paths(name), SESSIONS) if args.recorded else None
        dest = args.out or Path("out") / f"{name}.gif"
        print(json.dumps(rollout(args.export or Path("exports") / name, dest, MODELS[name].temperature, args.seconds, args.seed, recorded), indent=1))
    return 0
