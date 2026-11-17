from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_io import load_config, save_json  # noqa: E402


METHOD_IDS = ["backbone", "ft", "score"]
METRIC_KEYS = ["precision", "recall", "f1", "aupr", "auroc"]


def parse_seeds(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return raw
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seeds must contain at least one integer")
    return seeds


def strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if raw in {"null", "None", "~"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def load_registry(path: str) -> dict[str, Any]:
    registry_path = ROOT / path
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)

    try:
        import yaml  # type: ignore

        with registry_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for raw_line in registry_path.read_text(encoding="utf-8").splitlines():
        line = strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if text.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unsupported YAML list placement: {raw_line}")
            parent.append(parse_scalar(text[2:]))
            continue

        key, sep, value = text.partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
            continue
        child: Any = [] if key == "notes" else {}
        parent[key] = child
        stack.append((indent, child))
    return root


def method_meta(registry: dict[str, Any], method_id: str) -> dict[str, Any]:
    try:
        return registry["methods"][method_id]
    except KeyError as exc:
        raise KeyError(f"Missing method_id in registry: {method_id}") from exc


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_clean_targets(paths: list[Path]) -> None:
    existing = [p for p in paths if p.exists()]
    if existing:
        msg = "\n".join(f"- {rel(p)}" for p in existing)
        raise SystemExit(
            "Refusing to overwrite existing multiseed outputs. "
            "Move/remove these paths first or choose a different --output-root:\n"
            f"{msg}"
        )


def planned_paths(seed: int, output_root: Path) -> dict[str, Any]:
    seed_root = output_root / f"seed_{seed}"
    return {
        "seed_root": seed_root,
        "seed_json": seed_root / "seed_result.json",
        "methods": {
            method_id: {
                "out_dir": seed_root / method_id,
                "config": seed_root / method_id / "config.yaml",
                "ckpt": seed_root / method_id / "best.pt",
            }
            for method_id in METHOD_IDS
        },
    }


def write_config(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def make_seed_config(base_cfg: dict[str, Any], seed: int, out_dir: Path, suffix: str) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg.setdefault("protocol", {})
    cfg["protocol"]["name"] = f"{cfg['protocol'].get('name', 'swat')}_seed_{seed}_{suffix}"
    cfg["output"]["dir"] = str(out_dir.as_posix())
    return cfg


def make_score_config(ft_cfg: dict[str, Any], seed: int, out_dir: Path, alias: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(ft_cfg)
    cfg["seed"] = int(seed)
    cfg["output"]["dir"] = str(out_dir.as_posix())
    cfg["eval"] = {
        "alpha": float(args.score_alpha),
        "beta": float(args.score_beta),
        "gamma": float(args.score_gamma),
        "smooth_window": int(args.score_smooth_window),
        "threshold_mode": "pot",
        "pot_q": float(args.score_pot_q),
        "pot_target_prob": float(args.score_pot_target_prob),
    }
    cfg.setdefault("protocol", {})
    cfg["protocol"]["name"] = f"{cfg['protocol'].get('name', 'swat')}_seed_{seed}_score"
    cfg["protocol"]["notes"] = (
        f"{alias} does not train a new model. It evaluates the current seed FT checkpoint "
        "with fixed score-fusion parameters."
    )
    return cfg


def metrics_from_eval(result: dict[str, Any]) -> dict[str, float]:
    metrics = result["metrics"]
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def validate_registry_paths(registry: dict[str, Any]) -> list[str]:
    missing = []
    for method_id in METHOD_IDS:
        train_config = str(method_meta(registry, method_id).get("train_config", "TODO"))
        if train_config.startswith("TODO"):
            if method_id != "score":
                missing.append(f"{method_id}.train_config TODO: {train_config}")
            continue
        if not (ROOT / train_config).exists():
            missing.append(f"{method_id}.train_config missing path: {train_config}")
    return missing


def dry_run(seed: int, registry: dict[str, Any], args: argparse.Namespace, output_root: Path) -> None:
    paths = planned_paths(seed, output_root)
    print(f"=== Dry run seed {seed} ===")
    for method_id in METHOD_IDS:
        meta = method_meta(registry, method_id)
        alias = meta["paper_alias"]
        train_config = str(meta.get("train_config", "TODO"))
        if train_config.startswith("TODO"):
            config_status = f"TODO path: {train_config}"
        else:
            config_status = "exists" if (ROOT / train_config).exists() else f"missing path: {train_config}"
        method_paths = paths["methods"][method_id]
        print(f"[{method_id}] paper_alias={alias}")
        print(f"  train_config: {train_config} ({config_status})")
        print(f"  output_dir: {rel(method_paths['out_dir'])}")
        print(f"  config_out: {rel(method_paths['config'])}")
        if method_id == "backbone":
            print(f"  train: {sys.executable} train.py --config {rel(method_paths['config'])}")
            print(f"  checkpoint: {rel(method_paths['ckpt'])}")
            print(f"  eval: {sys.executable} evaluate.py --config {rel(method_paths['config'])} --ckpt {rel(method_paths['ckpt'])}")
        elif method_id == "ft":
            backbone_ckpt = paths["methods"]["backbone"]["ckpt"]
            print(
                "  train: "
                f"{sys.executable} train.py --config {rel(method_paths['config'])} --init-ckpt {rel(backbone_ckpt)}"
            )
            print(f"  init_checkpoint: {rel(backbone_ckpt)} (planned from current seed backbone)")
            print(f"  checkpoint: {rel(method_paths['ckpt'])}")
            print(f"  eval: {sys.executable} evaluate.py --config {rel(method_paths['config'])} --ckpt {rel(method_paths['ckpt'])}")
        else:
            ft_ckpt = paths["methods"]["ft"]["ckpt"]
            print("  train: <none; score does not train>")
            print(f"  checkpoint: {rel(ft_ckpt)} (planned from current seed ft)")
            print(f"  eval: {sys.executable} evaluate.py --config {rel(method_paths['config'])} --ckpt {rel(ft_ckpt)}")
    print(f"  seed_result_json: {rel(paths['seed_json'])}")


def run_seed(seed: int, registry: dict[str, Any], args: argparse.Namespace, output_root: Path) -> list[dict[str, Any]]:
    from evaluate import run_eval

    paths = planned_paths(seed, output_root)
    backbone_meta = method_meta(registry, "backbone")
    ft_meta = method_meta(registry, "ft")
    score_meta = method_meta(registry, "score")

    ensure_clean_targets(
        [
            paths["seed_root"],
        ]
    )

    backbone_cfg = make_seed_config(
        load_config(backbone_meta["train_config"]),
        seed,
        paths["methods"]["backbone"]["out_dir"],
        "backbone",
    )
    ft_cfg = make_seed_config(
        load_config(ft_meta["train_config"]),
        seed,
        paths["methods"]["ft"]["out_dir"],
        "ft",
    )
    score_cfg = make_score_config(
        ft_cfg,
        seed,
        paths["methods"]["score"]["out_dir"],
        score_meta["paper_alias"],
        args,
    )

    write_config(backbone_cfg, paths["methods"]["backbone"]["config"])
    write_config(ft_cfg, paths["methods"]["ft"]["config"])
    write_config(score_cfg, paths["methods"]["score"]["config"])

    run_command([sys.executable, "train.py", "--config", str(paths["methods"]["backbone"]["config"])])
    backbone_ckpt = paths["methods"]["backbone"]["ckpt"]
    if not backbone_ckpt.exists():
        raise FileNotFoundError(backbone_ckpt)

    run_command(
        [
            sys.executable,
            "train.py",
            "--config",
            str(paths["methods"]["ft"]["config"]),
            "--init-ckpt",
            str(backbone_ckpt),
        ]
    )
    ft_ckpt = paths["methods"]["ft"]["ckpt"]
    if not ft_ckpt.exists():
        raise FileNotFoundError(ft_ckpt)

    evals = {
        "backbone": run_eval(backbone_cfg, ckpt_path=str(backbone_ckpt)),
        "ft": run_eval(ft_cfg, ckpt_path=str(ft_ckpt)),
        "score": run_eval(score_cfg, ckpt_path=str(ft_ckpt)),
    }
    rows = []
    for method_id in METHOD_IDS:
        meta = method_meta(registry, method_id)
        ckpt = backbone_ckpt if method_id == "backbone" else ft_ckpt
        row = {
            "seed": int(seed),
            "method_id": method_id,
            "paper_alias": meta["paper_alias"],
            "checkpoint": str(ckpt.as_posix()),
            "config": str(paths["methods"][method_id]["config"].as_posix()),
            **metrics_from_eval(evals[method_id]),
        }
        rows.append(row)

    save_json(str(paths["seed_json"]), {"seed": int(seed), "rows": rows, "score_eval": score_cfg["eval"]})
    return rows


def write_raw_csv(rows: list[dict[str, Any]], output_root: Path) -> None:
    path = output_root / "multiseed_raw.csv"
    ensure_clean_targets([path])
    fieldnames = ["seed", "method_id", "paper_alias", *METRIC_KEYS, "checkpoint", "config"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_seed_raw_csv(seed: int, rows: list[dict[str, Any]], output_root: Path) -> None:
    path = output_root / "raw" / f"seed_{seed}.csv"
    ensure_clean_targets([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "method_id", "paper_alias", *METRIC_KEYS, "checkpoint", "config"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    path = output_root / "multiseed_summary.csv"
    ensure_clean_targets([path])
    out_rows = []
    for method_id in METHOD_IDS:
        method_rows = [r for r in rows if r["method_id"] == method_id]
        if not method_rows:
            continue
        out = {
            "method_id": method_id,
            "paper_alias": method_rows[0]["paper_alias"],
            "n_seeds": len(method_rows),
            "seeds": ",".join(str(r["seed"]) for r in method_rows),
        }
        for metric in METRIC_KEYS:
            values = [float(r[metric]) for r in method_rows]
            mu, sd = mean_std(values)
            out[f"{metric}_mean"] = f"{mu:.6f}"
            out[f"{metric}_std"] = f"{sd:.6f}"
            out[f"{metric}_mean_std"] = f"{mu:.6f}+/-{sd:.6f}"
        out_rows.append(out)

    fieldnames = ["method_id", "paper_alias", "n_seeds", "seeds"]
    for metric in METRIC_KEYS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWaT multiseed experiments from configs/experiment_registry.yaml."
    )
    parser.add_argument("--seeds", type=parse_seeds, default=[1, 2, 3], help="Comma-separated seeds, e.g. 1,2,3")
    parser.add_argument("--registry", default="configs/experiment_registry.yaml")
    parser.add_argument("--output-root", default="results/multiseed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-alpha", type=float, default=0.6)
    parser.add_argument("--score-beta", type=float, default=0.15)
    parser.add_argument("--score-gamma", type=float, default=0.2)
    parser.add_argument("--score-smooth-window", type=int, default=3)
    parser.add_argument("--score-pot-q", type=float, default=0.93)
    parser.add_argument("--score-pot-target-prob", type=float, default=0.005)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    output_root = ROOT / args.output_root
    missing = validate_registry_paths(registry)

    if args.dry_run:
        for seed in args.seeds:
            dry_run(seed, registry, args, output_root)
        if missing:
            print("Missing registry paths:")
            for item in missing:
                print(f"- {item}")
        return

    if missing:
        raise SystemExit("Missing registry paths:\n" + "\n".join(f"- {item}" for item in missing))

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_clean_targets([output_root / "multiseed_raw.csv", output_root / "multiseed_summary.csv"])

    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"=== Running seed {seed} ===")
        seed_rows = run_seed(seed, registry, args, output_root)
        write_seed_raw_csv(seed, seed_rows, output_root)
        all_rows.extend(seed_rows)

    write_raw_csv(all_rows, output_root)
    write_summary(all_rows, output_root)
    print(f"Saved raw CSV: {rel(output_root / 'multiseed_raw.csv')}")
    print(f"Saved summary: {rel(output_root / 'multiseed_summary.csv')}")


if __name__ == "__main__":
    main()
