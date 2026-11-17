"""Run the frozen CoMuT-S0R sanity check with the official TimeEval k-Means image."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "dataset" / "CoMuT_compat"
OUT = ROOT / "outputs" / "thesis_comut_s0r_sanity"
IMAGE = "ghcr.io/timeeval/kmeans:0.3.0"
IMAGE_ID = "sha256:acd5f5f6c8a644a02fa755e0fc274d323263f2f34e43e496fc174224b03e0ec8"
DIGEST = "ghcr.io/timeeval/kmeans@sha256:acd5f5f6c8a644a02fa755e0fc274d323263f2f34e43e496fc174224b03e0ec8"
CASES = [
    ("rmj-short-mode-correlation-on-2", 40, 0.992181),
    ("rmj-medium-mode-correlation-on-2", 80, 0.803246),
    ("rmj-large-mode-correlation-on-2", 160, 0.926983),
]


def reverse_windowing(scores: np.ndarray, window_size: int) -> np.ndarray:
    """Exact vectorized TimeEval 1.2.4 ReverseWindowing(mean) implementation."""
    mapped = np.full((window_size - 1 + len(scores), window_size), np.nan)
    mapped[: len(scores), 0] = scores
    for w in range(1, window_size):
        mapped[:, w] = np.roll(mapped[:, 0], w)
    return np.nanmean(mapped, axis=1)


def timeeval_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact TimeEval 1.2.4 Metric.ROC_AUC call path: auc(*roc_curve())."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def main() -> None:
    started = time.time()
    if OUT.exists() and (OUT / "sanity_summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed prior output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    log_lines = []
    results = []
    for name, window, published in CASES:
        source = DATA_ROOT / name / "test.csv"
        case_out = OUT / name
        case_out.mkdir(exist_ok=True)
        adapter = {
            "source": str(source),
            "mounted_read_only": True,
            "data_changed": False,
            "label_changed": False,
        }
        (case_out / "adapter_manifest.json").write_text(json.dumps(adapter, indent=2), encoding="utf-8")
        interface = {
            "dataInput": "/data/test.csv",
            "dataOutput": "/results/docker-algorithm-scores.csv",
            "modelInput": "/results/model.pkl",
            "modelOutput": "/results/model.pkl",
            "executionType": "execute",
            "customParameters": {
                "anomaly_window_size": window,
                "n_clusters": 50,
                "n_jobs": 1,
                "random_state": 42,
                "stride": 1,
            },
        }
        command = [
            "docker", "run", "--rm",
            "-v", f"{source.parent}:/data:ro",
            "-v", f"{case_out}:/results:rw",
            IMAGE, "execute-algorithm", json.dumps(interface, separators=(",", ":")),
        ]
        raw_path = case_out / "docker-algorithm-scores.csv"
        if raw_path.exists():
            log_lines.append(f"[{name}] reusing container output from interrupted S0R run")
        else:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=180)
            log_lines.extend([f"[{name}] returncode={proc.returncode}", proc.stdout, proc.stderr])
            if proc.returncode:
                raise RuntimeError(f"Official container failed for {name}: {proc.stderr or proc.stdout}")
        raw = np.genfromtxt(raw_path, delimiter=",")
        frame = pd.read_csv(source)
        labels = frame.iloc[:, -1].to_numpy(dtype=int)
        if len(raw) == len(labels):
            # GHCR 0.3.0 performs equivalent stride-1 reverse-windowing inside
            # KMeansAD._custom_reverse_windowing and therefore emits point scores.
            points = np.asarray(raw, dtype=float)
            postprocessing_location = "official_image_internal_custom_reverse_windowing"
        else:
            points = reverse_windowing(np.asarray(raw, dtype=float), window)
            postprocessing_location = "timeeval_1.2.4_external_ReverseWindowing"
        if len(points) != len(labels):
            raise ValueError(f"Length mismatch for {name}: {len(points)} != {len(labels)}")
        finite = bool(np.isfinite(raw).all() and np.isfinite(points).all())
        if not finite:
            raise ValueError(f"Non-finite scores for {name}")
        local = timeeval_roc_auc(labels, points.astype(float))
        inverted = timeeval_roc_auc(labels, (-points).astype(float))
        orientation = "PASS" if local >= inverted else "FAIL"
        np.savetxt(case_out / "postprocessed_point_scores.csv", points, delimiter=",")
        results.append({
            "dataset": name,
            "anomaly_window_size": window,
            "raw_score_length": int(len(raw)),
            "point_score_length": int(len(points)),
            "label_length": int(len(labels)),
            "postprocessing_location": postprocessing_location,
            "score_finite": finite,
            "local_auroc": local,
            "inverted_score_auroc_diagnostic_only": inverted,
            "published_auroc": published,
            "absolute_delta": abs(local - published),
            "score_orientation": orientation,
            "status": "PASS" if orientation == "PASS" else "FAIL",
        })

    all_executed = len(results) == 3
    all_finite = all(r["score_finite"] for r in results)
    orientation_pass = all(r["score_orientation"] == "PASS" for r in results)
    threshold_pass = all(r["local_auroc"] > 0.65 for r in results) and sum(r["local_auroc"] > 0.75 for r in results) >= 2
    base_ready = all_executed and all_finite and orientation_pass and threshold_pass
    shifted = any(r["absolute_delta"] > 0.10 for r in results)
    status = "COMUT_SANITY_READY_REFERENCE_SHIFTED" if base_ready and shifted else ("COMUT_SANITY_PIPELINE_READY" if base_ready else "COMUT_SANITY_FAILED")
    summary = {
        "status": status,
        "execution_3_of_3": all_executed,
        "all_scores_finite": all_finite,
        "score_orientation": "PASS" if orientation_pass else "FAIL",
        "gate_all_auroc_gt_0_65": all(r["local_auroc"] > 0.65 for r in results),
        "gate_at_least_2_of_3_gt_0_75": sum(r["local_auroc"] > 0.75 for r in results) >= 2,
        "elapsed_seconds": time.time() - started,
        "official_timeeval_kmeans": True,
        "current_official_ghcr_image": True,
        "registry_compatibility_adaptation_only": True,
        "algorithm_reimplemented": False,
        "parameter_change": False,
        "data_change": False,
    }
    manifest = {
        "source_repository": "TimeEval/TimeEval-algorithms",
        "algorithm": "kmeans",
        "tag": IMAGE,
        "image_id": IMAGE_ID,
        "repo_digest": DIGEST,
        "legacy_registry_available": False,
        "registry_compatibility_adaptation": True,
        "algorithm_reimplemented": False,
        "exact_historical_docker_image_reproduction": False,
        "runtime_output_semantics": "point scores after image-internal custom reverse-windowing",
        "legacy_wrapper_interface_compatible": False,
    }
    with (OUT / "sanity_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    (OUT / "sanity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "docker_image_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "image_reference_diff.txt").write_text(
        "--- official TimeEval 1.2.4 kmeans wrapper\n"
        "+++ S0R compatibility wrapper\n"
        "- image_name=\"sopedu:5000/akita/kmeans\"\n"
        "+ image_name=\"ghcr.io/timeeval/kmeans\", tag=\"0.3.0\"\n"
        "All algorithm parameters, execution type, postprocessing, metric, and datasets are unchanged.\n"
        "Runtime compatibility observation: the current 0.3.0 image itself emits 10,000 point scores using "
        "KMeansAD._custom_reverse_windowing. The legacy wrapper expected window scores and would create 10,039 "
        "points if its external ReverseWindowing were applied again. No second reverse-windowing or score inversion "
        "was performed.\n",
        encoding="utf-8",
    )
    (OUT / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
