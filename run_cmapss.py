from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cmapss.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_train:
        run([py, "train.py", "--config", args.config])
    if not args.skip_eval:
        run([py, "evaluate.py", "--config", args.config])


if __name__ == "__main__":
    main()
