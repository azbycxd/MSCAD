from __future__ import annotations

from data.cmapss import build_cmapss_dataloaders
from data.hai import build_hai_dataloaders
from data.smap import build_smap_dataloaders
from data.swat import build_swat_dataloaders


def build_dataloaders(cfg: dict, pin_memory: bool = True):
    name = str(cfg.get("dataset", "swat")).lower()
    data_cfg = cfg["data"]
    if name == "swat":
        return build_swat_dataloaders(
            train_csv=data_cfg["train_csv"],
            test_csv=data_cfg["test_csv"],
            label_col=data_cfg["label_col"],
            timestamp_col=data_cfg["timestamp_col"],
            window_size=int(data_cfg["window_size"]),
            stride_train=int(data_cfg["stride_train"]),
            stride_test=int(data_cfg["stride_test"]),
            val_ratio=float(data_cfg.get("val_ratio", 0.2)),
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=pin_memory,
        )
    if name == "hai":
        return build_hai_dataloaders(
            data_root=data_cfg["data_root"],
            window_size=int(data_cfg["window_size"]),
            stride_train=int(data_cfg["stride_train"]),
            stride_test=int(data_cfg["stride_test"]),
            validation_ratio=float(data_cfg.get("val_ratio", 0.2)),
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=pin_memory,
            seed=int(cfg.get("seed", 42)),
        )
    if name == "smap":
        return build_smap_dataloaders(
            train_path=data_cfg.get("train_path"),
            test_path=data_cfg.get("test_path"),
            test_label_path=data_cfg.get("test_label_path"),
            omni_dir=data_cfg.get("omni_dir"),
            channel=data_cfg.get("channel"),
            labeled_anomalies_csv=data_cfg.get("labeled_anomalies_csv"),
            spacecraft=data_cfg.get("spacecraft", "SMAP"),
            window_size=int(data_cfg["window_size"]),
            stride_train=int(data_cfg["stride_train"]),
            stride_test=int(data_cfg["stride_test"]),
            val_ratio=float(data_cfg.get("val_ratio", 0.2)),
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=pin_memory,
        )
    if name == "cmapss":
        return build_cmapss_dataloaders(
            source=str(data_cfg.get("source", "npz")),
            npz_path=data_cfg.get("npz_path"),
            train_path=data_cfg.get("train_path"),
            test_path=data_cfg.get("test_path"),
            rul_path=data_cfg.get("rul_path"),
            label_mode=str(data_cfg.get("label_mode", "last_k")),
            last_k_cycles=int(data_cfg.get("last_k_cycles", 30)),
            rul_threshold=int(data_cfg.get("rul_threshold", 30)),
            use_ops=bool(data_cfg.get("use_ops", True)),
            use_sensors=bool(data_cfg.get("use_sensors", True)),
            near_const_var_threshold=float(data_cfg.get("near_const_var_threshold", 1e-8)),
            window_size=int(data_cfg["window_size"]),
            stride_train=int(data_cfg["stride_train"]),
            stride_test=int(data_cfg["stride_test"]),
            val_ratio=float(data_cfg.get("val_ratio", 0.2)),
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=pin_memory,
        )
    raise ValueError(f"Unsupported dataset: {cfg.get('dataset')}")
