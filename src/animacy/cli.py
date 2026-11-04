"""animacy: fetch the recordings, build the corpus, train, export for the browser, and draw a rollout.

    animacy fetch fish        download the fish sessions (flies, both: the fly sessions or all)
    animacy build fish        the corpus and scenes in data/fish/
    animacy train fish        the three training stages in runs/fish/
    animacy export fish       exports/fish/ from the last stage
    animacy rollout fish      out/fish.gif: the exported model continuing a held-out school
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

    recipe = MODELS[name]
    records, runs, zooms = [], [], {}
    for family in recipe.families:
        r, f = data.family_runs(family, SESSIONS)
        if recipe.body_lengths:
            f, z = data.in_body_lengths(f, SESSIONS)
            zooms |= z
            # A tracking region is in pixels, wrong once positions are rescaled. It is dropped rather than
            # rescaled because the shipped scenes.json and seeds' arenas were built without it.
            r = [{**x, "roi": None} for x in r]
        records += r
        runs += f
    corpus_dir, scenes_dir = paths(name)
    summary = corpus.build(runs, corpus_dir)
    if zooms:
        (corpus_dir / "zooms.json").write_text(json.dumps(zooms, indent=1), encoding="utf-8", newline="\n")
    built = scenes.build(runs, records, corpus_dir, scenes_dir)
    print(json.dumps({"scale": summary["scale"], "corpus": {p: a["sequences"] for p, a in summary["arrays"].items()}, "scenes": built["partitions"]}, indent=1))


def train(name: str, stage: str | None, steps: int | None, batch: int, out: Path | None) -> None:
    import torch

    from . import train

    corpus_dir, scenes_dir = paths(name)
    previous = None
    for s in STAGES:
        here = (out or Path("runs") / name) / s.name
        if stage in (None, s.name):
            wanted = {"steps": steps or s.steps, "lr": s.lr, "mean_weight": s.mean_weight, "batch": batch}
            if (here / "model.pt").exists():
                run = torch.load(here / "model.pt", map_location="cpu", weights_only=False)["run"]
                if any(run.get(k) != v for k, v in wanted.items()):
                    raise SystemExit(f"{here} was trained with {({k: run.get(k) for k in wanted})}, not {wanted}: remove it or pass --out")
                print(f"{name}: {s.name} already trained in {here}, skipped", flush=True)
            else:
                print(f"{name}: {s.name}", flush=True)
                train.train(corpus_dir, scenes_dir if s.neighbours else None, here, wanted["steps"], s.lr, s.mean_weight, previous, batch)
        if stage == s.name:
            return
        previous = here / "model.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animacy", description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog="\n".join(__doc__.splitlines()[2:]))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "build", "train", "export", "rollout"):
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
            sub.add_argument("--export", type=Path, help="exports/<model> by default")
            sub.add_argument("--seconds", type=float, default=20.0)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--out", type=Path, help="out/<model>.gif by default")
    args = parser.parse_args(argv)
    name = args.model
    if args.command == "fetch":
        return fetch(name)
    elif args.command == "build":
        build(name)
    elif args.command == "train":
        train(name, args.stage, args.steps, args.batch, args.out)
    elif args.command == "export":
        import torch

        from .export import export

        corpus_dir, scenes_dir = paths(name)
        checkpoint = args.checkpoint or Path("runs") / name / STAGES[-1].name / "model.pt"
        step = torch.load(checkpoint, map_location="cpu", weights_only=False)["step"]
        if args.out is None and step != STAGES[-1].steps:
            raise SystemExit(f"{checkpoint} was trained for {step} steps, not {STAGES[-1].steps}: pass --out to export it outside exports/")
        print(json.dumps(export(checkpoint, corpus_dir, scenes_dir, SESSIONS, args.out or Path("exports") / name, MODELS[name].head_windows), indent=1))
    elif args.command == "rollout":
        from .rollout import rollout

        corpus_dir, scenes_dir = paths(name)
        dest = args.out or Path("out") / f"{name}.gif"
        print(json.dumps(rollout(args.export or Path("exports") / name, corpus_dir, scenes_dir, SESSIONS, dest, MODELS[name].temperature, args.seconds, args.seed), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
