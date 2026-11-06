"""Command line: ``sway-eval audit | generate | plan | run | analyze | report``.

    sway-eval audit CONFIG [--yes]   generate the records if needed, plan, run, analyze and
                                     write report.md and report.json in the run folder

    sway-eval generate CONFIG        write the processed items the config names
    sway-eval plan CONFIG            print planned calls, cost and wall time
    sway-eval run CONFIG [--yes]     run (paid or >1 h runs need --yes); exit 3 if a canary halted it
    sway-eval analyze RUN_DIR        write per_item.jsonl and summary.json
    sway-eval report RUN_DIR         write report.md and report.json for a finished run
"""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import analyze
from .config import build_adapter, load_config, resolve, rubric_spec
from .records import load_items, sample_items, save_items
from .report import write_report
from .runner import execute, make_plan
from .synth import generate


def _items(cfg: dict):
    path = resolve(cfg, cfg["dataset"]["items"])
    if not path.exists():
        raise SystemExit(f"{path} not found; run `sway-eval generate` first")
    items = load_items(path)
    sub = cfg["dataset"].get("sample")  # {n, seed, stratify}: a seeded subsample of the items
    return sample_items(items, sub["n"], sub["seed"], sub.get("stratify")) if sub else items


def cmd_generate(cfg: dict) -> None:
    """Build the processed items a config names: synthetic, or from raw files."""
    d = cfg["dataset"]
    if d.get("kind") == "synthetic_rubric":
        items = generate(rubric_spec(cfg), list(range(d["levels"][0], d["levels"][1] + 1)), d["per_level"], cfg["seed"])
    elif d.get("kind") == "climate_fever":
        from .loaders import climate_fever
        items = climate_fever.load(resolve(cfg, d["raw"]))
    else:
        raise SystemExit(f"generate can't build dataset kind {d.get('kind')!r}")
    path = resolve(cfg, d["items"])
    if path.exists():
        raise SystemExit(f"{path} exists; processed data are regenerated only into a new path")
    n = save_items(items, path)
    print(f"Wrote {n} items to {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sway-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "generate", "plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("config")
        if name in ("audit", "run"):
            p.add_argument("--yes", action="store_true", help="confirm a paid or long run")
    for name in ("analyze", "report"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        print(json.dumps(analyze(args.run_dir), indent=2, default=str))
        return 0
    if args.cmd == "report":
        print(write_report(args.run_dir).read_text(encoding="utf-8"))
        return 0
    cfg = load_config(args.config)
    if args.cmd == "audit":
        if not resolve(cfg, cfg["dataset"]["items"]).exists():
            cmd_generate(cfg)
        out = execute(_items(cfg), cfg, build_adapter(cfg), yes=args.yes)
        if not out:
            return 2
        md = write_report(out)
        print(md.read_text(encoding="utf-8"))
        print(f"Report: {md}")
        return 3 if json.loads((out / "run.json").read_text(encoding="utf-8")).get("halted") else 0
    if args.cmd == "generate":
        cmd_generate(cfg)
    elif args.cmd == "plan":
        adapter = build_adapter(cfg)
        print(make_plan(_items(cfg), cfg, paid=adapter.paid).text())
    else:
        out = execute(_items(cfg), cfg, build_adapter(cfg), yes=args.yes)
        if not out:
            return 2
        return 3 if json.loads((out / "run.json").read_text(encoding="utf-8")).get("halted") else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
