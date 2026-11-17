from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from evaluate import run_eval
from train import main as train_main  # noqa: F401
from utils.config_io import load_config, save_json

import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def build_variants(base_cfg: dict) -> list[tuple[str, dict]]:
    v = []

    def pack(name: str, updater):
        cfg = copy.deepcopy(base_cfg)
        updater(cfg)
        run_name = f"ablation_main_{name}"
        cfg["output"]["dir"] = f"outputs/{run_name}"
        v.append((run_name, cfg))

    pack("full", lambda c: c)
    pack("wo_short", lambda c: c["train"].update({"lambda_s": 0.0}))
    pack("wo_long", lambda c: c["train"].update({"lambda_l": 0.0}))
    pack(
        "wo_struct",
        lambda c: c["train"].update({"lambda_s": 0.0, "lambda_l": 0.0}),
    )
    pack("cov", lambda c: c["train"].update({"cov_type": "cov"}))
    pack("corr", lambda c: c["train"].update({"cov_type": "corr"}))
    pack("shrink_cov", lambda c: c["train"].update({"cov_type": "shrink_cov"}))
    pack("mean3std", lambda c: c["eval"].update({"threshold_mode": "mean3std"}))
    return v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swat_main.yaml")
    parser.add_argument("--only", default=None, help="Run only one variant by name.")
    parser.add_argument("--resume", action="store_true", help="Skip variants with existing eval_metrics.json.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    variants = build_variants(base_cfg)
    if args.only:
        variants = [x for x in variants if x[0] == args.only]
        if not variants:
            raise ValueError(f"Unknown variant: {args.only}")

    summary = {}
    py = sys.executable
    for name, cfg in variants:
        out_dir = Path(cfg["output"]["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_path = out_dir / "eval_metrics.json"
        if args.resume and eval_path.exists():
            print(f"\n=== Skipping {name} (exists) ===")
            summary[name] = load_config(str(eval_path))
            continue
        cfg_path = out_dir / "config.yaml"
        save_json(str(cfg_path), cfg)
        print(f"\n=== Running {name} ===")
        run_cmd([py, "train.py", "--config", str(cfg_path)])
        result = run_eval(cfg)
        summary[name] = result
        print(f"{name}: F1={result['metrics']['f1']:.4f}, AUPR={result['metrics']['aupr']:.4f}")

    save_json("outputs/ablation_main_summary.json", summary)
    print("Ablation completed: outputs/ablation_main_summary.json")


if __name__ == "__main__":
    main()
