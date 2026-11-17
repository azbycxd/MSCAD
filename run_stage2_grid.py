from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

from utils.config_io import load_config, save_json


def run_cmd(cmd: List[str], cwd: str) -> None:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def read_eval(eval_path: Path) -> Dict[str, Any]:
    return load_config(str(eval_path))


def build_cfg(base_cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    c = copy.deepcopy(base_cfg)
    c["output"]["dir"] = f"outputs/stage2_grid/{name}"
    return c


def write_cfg(cfg: Dict[str, Any], path: Path) -> None:
    save_json(str(path), cfg)


def run_train_eval(cwd: str, py: str, cfg_path: Path, init_ckpt: str | None = None) -> Dict[str, Any]:
    train_cmd = [py, "train.py", "--config", str(cfg_path)]
    if init_ckpt is not None:
        train_cmd.extend(["--init-ckpt", init_ckpt])
    run_cmd(train_cmd, cwd=cwd)
    run_cmd([py, "evaluate.py", "--config", str(cfg_path)], cwd=cwd)
    cfg = load_config(str(cfg_path))
    eval_path = Path(cwd) / cfg["output"]["dir"] / "eval_metrics.json"
    train_path = Path(cwd) / cfg["output"]["dir"] / "train_metrics.json"
    ev = read_eval(eval_path)
    tr = read_eval(train_path)
    return {"eval": ev, "train": tr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/swat_main.yaml")
    parser.add_argument("--workdir", default="D:/workspace/python/mscad")
    args = parser.parse_args()

    cwd = args.workdir
    py = sys.executable
    base = load_config(args.base_config)
    root = Path(cwd) / "outputs" / "stage2_grid"
    root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"runs": {}}

    # Stage A0: backbone wo_struct
    a0 = build_cfg(base, "A0_backbone_wo_struct")
    a0["train"]["lambda_s"] = 0.0
    a0["train"]["lambda_l"] = 0.0
    a0["train"]["structure_warmup_epochs"] = 0
    a0["train"]["structure_pairs"] = {"enabled": False}
    a0["eval"]["alpha"] = 1.0
    a0["eval"]["beta"] = 0.0
    a0["eval"]["gamma"] = 0.0
    a0["train"]["epochs"] = min(int(a0["train"]["epochs"]), 20)
    a0["train"]["patience"] = min(int(a0["train"]["patience"]), 5)
    a0_path = root / "A0_backbone_wo_struct_config.yaml"
    write_cfg(a0, a0_path)
    out = run_train_eval(cwd, py, a0_path, init_ckpt=None)
    summary["runs"]["A0_backbone_wo_struct"] = out
    backbone_ckpt = str((Path(cwd) / a0["output"]["dir"] / "best.pt").resolve())

    # Group A
    group_a = [
        ("A1_ls0.05_ll0.00_top20_corr0.9_ws64_ss16", 0.05, 0.00, 20, 0.9, 64, 16),
        ("A2_ls0.05_ll0.02_top20_corr0.9_ws64_ss16", 0.05, 0.02, 20, 0.9, 64, 16),
        ("A3_ls0.10_ll0.02_top20_corr0.9_ws64_ss16", 0.10, 0.02, 20, 0.9, 64, 16),
        ("A4_ls0.10_ll0.05_top20_corr0.9_ws64_ss16", 0.10, 0.05, 20, 0.9, 64, 16),
    ]
    best_a_name = None
    best_a_f1 = -1.0
    best_a_cfg = None
    for name, ls, ll, topk, mc, ws, ss in group_a:
        c = build_cfg(base, name)
        c["data"]["window_size"] = ws
        c["data"]["short_window"] = ss
        c["train"]["lambda_s"] = ls
        c["train"]["lambda_l"] = ll
        c["train"]["structure_warmup_epochs"] = 10
        c["train"]["structure_pairs"] = {"enabled": True, "topk": topk, "min_abs_corr": mc}
        c["eval"]["alpha"] = 0.5
        c["eval"]["beta"] = 0.25
        c["eval"]["gamma"] = 0.25
        c["train"]["epochs"] = min(int(c["train"]["epochs"]), 20)
        c["train"]["patience"] = min(int(c["train"]["patience"]), 5)
        p = root / f"{name}_config.yaml"
        write_cfg(c, p)
        out = run_train_eval(cwd, py, p, init_ckpt=backbone_ckpt)
        summary["runs"][name] = out
        f1 = float(out["eval"]["metrics"]["f1"])
        if f1 > best_a_f1:
            best_a_f1 = f1
            best_a_name = name
            best_a_cfg = c

    assert best_a_cfg is not None
    summary["best_A"] = {"name": best_a_name, "f1": best_a_f1}

    # Group B (pair scale)
    group_b = [
        ("B1_top40_corr0.8", 40, 0.8),
        ("B2_top20_corr0.8", 20, 0.8),
    ]
    for tag, topk, mc in group_b:
        name = f"{tag}_from_{best_a_name}"
        c = copy.deepcopy(best_a_cfg)
        c["output"]["dir"] = f"outputs/stage2_grid/{name}"
        c["train"]["structure_pairs"] = {"enabled": True, "topk": topk, "min_abs_corr": mc}
        p = root / f"{name}_config.yaml"
        write_cfg(c, p)
        out = run_train_eval(cwd, py, p, init_ckpt=backbone_ckpt)
        summary["runs"][name] = out

    # Group C (scale)
    group_c = [
        ("C1_ws48_ss12", 48, 12),
        ("C2_ws96_ss24", 96, 24),
    ]
    for tag, ws, ss in group_c:
        name = f"{tag}_from_{best_a_name}"
        c = copy.deepcopy(best_a_cfg)
        c["output"]["dir"] = f"outputs/stage2_grid/{name}"
        c["data"]["window_size"] = ws
        c["data"]["short_window"] = ss
        p = root / f"{name}_config.yaml"
        write_cfg(c, p)
        out = run_train_eval(cwd, py, p, init_ckpt=backbone_ckpt)
        summary["runs"][name] = out

    # Flat summary table
    rows = []
    for name, out in summary["runs"].items():
        m = out["eval"]["metrics"]
        rows.append(
            {
                "name": name,
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "aupr": float(m["aupr"]),
                "auroc": float(m["auroc"]),
            }
        )
    rows = sorted(rows, key=lambda x: x["f1"], reverse=True)
    summary["ranked"] = rows
    save_json(str(root / "stage2_grid_summary.json"), summary)
    print(f"Saved: {root / 'stage2_grid_summary.json'}")
    print("Top runs:")
    for r in rows[:5]:
        print(f"{r['name']}: F1={r['f1']:.4f}, AUPR={r['aupr']:.4f}")


if __name__ == "__main__":
    main()
