from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.hai import HAISequence, MultiSequenceWindowDataset, build_hai_dataloaders, extract_attack_events
from data.transforms import StandardScaler


FEATURES = [f"F{i:02d}" for i in range(86)]


def sequence(name: str, length: int, split: str = "train", offset: int = 0) -> HAISequence:
    x = np.arange(length * 3, dtype=np.float32).reshape(length, 3) + offset
    y = np.zeros(length, dtype=np.int64)
    y[10:20] = 1
    timestamps = np.asarray([f"2021-01-01 00:{i // 60:02d}:{i % 60:02d}" for i in range(length)])
    return HAISequence(name, f"{name}.csv", split, x, y, timestamps)


@pytest.fixture(scope="module")
def hai_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("hai")
    rows = 800
    for file_index, name in enumerate([f"train{i}" for i in range(1, 7)] + [f"test{i}" for i in range(1, 5)]):
        timestamps = pd.date_range("2021-01-01", periods=rows, freq="s") + pd.Timedelta(days=file_index)
        values = np.arange(rows, dtype=np.float32)[:, None] + np.arange(86, dtype=np.float32)[None, :] + file_index * 10
        values[:, -1] = 5.0
        if name.startswith("train"):
            values[640:] += 10_000
            labels = np.zeros(rows, dtype=np.int64)
        else:
            values += 20_000
            labels = np.zeros(rows, dtype=np.int64)
            labels[100:120] = 1
            labels[400:430] = 1
        frame = pd.DataFrame(values, columns=FEATURES)
        frame.insert(0, "timestamp", timestamps.strftime("%Y-%m-%d %H:%M:%S"))
        frame["Attack"] = labels
        frame.to_csv(root / f"{name}.csv", index=False)
    return root


@pytest.fixture(scope="module")
def bundle64(hai_root):
    return build_hai_dataloaders(hai_root, window_size=64, batch_size=16, seed=7)


def test_six_sequences_never_cross_files(bundle64):
    dataset = bundle64.train_loader.dataset
    assert len(dataset.sequences) == 6
    assert all(count == (640 - 64 + 1) for count in dataset.window_counts)


def test_train_validation_do_not_cross_split(bundle64):
    assert all(sequence.original_row_offset == 0 for sequence in bundle64.train_sequences)
    assert all(sequence.original_row_offset == 640 for sequence in bundle64.val_sequences)
    assert all(len(sequence.x) == 160 for sequence in bundle64.val_sequences)


def test_global_index_mapping():
    ds = MultiSequenceWindowDataset([sequence("a", 70), sequence("b", 80)], window_size=64)
    assert ds.locate_index(0) == (0, 0)
    assert ds.locate_index(7) == (1, 0)
    assert ds.locate_index(len(ds) - 1) == (1, 16)


def test_first_and_last_windows():
    seq = sequence("a", 70)
    ds = MultiSequenceWindowDataset([seq], window_size=64)
    assert torch.equal(ds[0][0], torch.from_numpy(seq.x[:64]))
    assert torch.equal(ds[-1][0], torch.from_numpy(seq.x[-64:]))


def test_label_uses_window_end():
    seq = sequence("a", 70)
    seq.y[:] = 0
    seq.y[63] = 1
    ds = MultiSequenceWindowDataset([seq], window_size=64)
    assert ds[0][1].item() == 1
    assert ds[1][1].item() == 0


def test_timestamp_matches_window_end():
    seq = sequence("a", 70)
    ds = MultiSequenceWindowDataset([seq], window_size=64, return_metadata=True)
    assert ds[0][2]["timestamp"] == seq.timestamps[63]
    assert ds[0][2]["end_row"] == 63


def test_window_64_and_128_counts(hai_root, bundle64):
    bundle128 = build_hai_dataloaders(hai_root, window_size=128, batch_size=16)
    assert len(bundle64.train_loader.dataset) == 6 * (640 - 64 + 1)
    assert len(bundle128.train_loader.dataset) == 6 * (640 - 128 + 1)


def test_scaler_only_fits_train_prefix(bundle64):
    expected_feature0_mean = np.mean([np.arange(640) + index * 10 for index in range(6)])
    assert np.isclose(bundle64.scaler.mean_[0, 0], expected_feature0_mean)
    assert bundle64.scaler.mean_[0, 0] < 1_000


def test_validation_and_test_do_not_affect_scaler(bundle64):
    assert float(bundle64.scaler.mean_.max()) < 1_000


def test_zero_variance_safe_scale():
    scaler = StandardScaler(eps=1e-8).fit_many([np.asarray([[1.0, 2.0], [1.0, 4.0]], dtype=np.float32)])
    transformed = scaler.transform(np.asarray([[1.0, 3.0]], dtype=np.float32))
    assert scaler.effective_scale_[0, 0] == 1.0
    assert np.isfinite(transformed).all()


def test_all_86_features_retained(bundle64):
    assert bundle64.feature_names == FEATURES


def test_test_files_are_separate(bundle64):
    assert list(bundle64.test_loaders) == ["test1", "test2", "test3", "test4"]
    assert all(len(loader.dataset.sequences) == 1 for loader in bundle64.test_loaders.values())


def test_attack_events_do_not_merge_files():
    a = sequence("test1", 70, "test")
    b = sequence("test2", 70, "test")
    a.y[:] = b.y[:] = 0
    a.y[-2:] = 1
    b.y[:2] = 1
    events = extract_attack_events([a, b])
    assert len(events) == 2
    assert events[0]["file_name"] != events[1]["file_name"]


def test_shuffle_reproducible(hai_root):
    a = build_hai_dataloaders(hai_root, window_size=64, batch_size=8, seed=123)
    b = build_hai_dataloaders(hai_root, window_size=64, batch_size=8, seed=123)
    assert torch.equal(next(iter(a.train_loader))[0], next(iter(b.train_loader))[0])


def test_default_batch_compatible(bundle64):
    batch = next(iter(bundle64.train_loader))
    assert len(batch) == 2
    assert batch[0].shape[1:] == (64, 86)


def test_metadata_contains_complete_index_information():
    seq = sequence("test1", 70, "test")
    seq.original_row_offset = 100
    ds = MultiSequenceWindowDataset([seq], 64, return_metadata=True)
    _, y, metadata = ds[0]
    assert set(metadata) == {"sequence_id", "file_name", "start_row", "end_row", "timestamp", "split", "original_label"}
    assert metadata["start_row"] == 100 and metadata["end_row"] == 163
    assert metadata["original_label"] == y.item()
