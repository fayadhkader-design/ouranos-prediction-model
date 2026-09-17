"""Train on earlier nominal windows, stop on validation loss, freeze checkpoint.
No test labels or test scores are accessed during fitting/calibration.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn

from src.neural.config import DEFAULT
from src.neural.network import WindowAutoencoder, window_batch
from src.neural.prepare import valid_window_ends


def nominal_rows(values, labels):
    return np.isfinite(values).all(axis=1) & (labels == 0).all(axis=1)


def fit_scaler(values, allowed, maximum=500000, seed=2026):
    indices = np.flatnonzero(allowed)
    if not len(indices):
        raise ValueError("No nominal training rows")
    if len(indices) > maximum:
        indices = np.random.default_rng(seed).choice(indices, maximum, replace=False)
    rows = values[indices].astype(np.float64)
    center = np.median(rows, axis=0)
    q25, q75 = np.quantile(rows, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    # Constant-IQR channels use standard deviation; do not divide by zero.
    scale = np.where(scale > 1e-6, scale, rows.std(axis=0))
    scale = np.maximum(scale, 1e-6)
    return center.astype(np.float32), scale.astype(np.float32)


def choose_windows(ends, maximum, rng):
    return rng.choice(ends, min(maximum, len(ends)), replace=False)


def checkpoint_model(checkpoint):
    cfg = checkpoint["config"]
    model = WindowAutoencoder(
        len(cfg["channels"]), cfg["window"], cfg["hidden"], cfg["bottleneck"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def train(
    data=Path("data/processed/esa_neural"),
    output=Path("models/esa_neural"),
    reports=Path("reports/esa_neural"),
    config=DEFAULT,
):
    data, output, reports = Path(data), Path(output), Path(reports)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((data / "preparation.json").read_text())
    if preparation["source"] != "REAL MISSION DATA":
        raise ValueError("Real-data provenance required")
    if preparation["config"] != json.loads(json.dumps(config.to_dict())):
        raise ValueError(
            "Prepared dataset configuration differs; re-prepare before training"
        )
    values = np.load(data / "values.npy", mmap_mode="r")
    labels = np.load(data / "labels.npy", mmap_mode="r")
    splits = np.load(data / "splits.npy", mmap_mode="r")
    nominal = nominal_rows(values, labels)
    train_mask = nominal & (splits == 0)
    validation_mask = nominal & (splits == 1)
    train_ends = valid_window_ends(train_mask, config.window, config.train_stride)
    validation_ends = valid_window_ends(
        validation_mask, config.window, config.train_stride
    )
    if len(train_ends) < 1000 or len(validation_ends) < 100:
        raise ValueError(
            "Insufficient complete nominal windows for chronological training/validation"
        )
    mean, scale = fit_scaler(values, train_mask, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    training = choose_windows(train_ends, config.max_train_windows, rng)
    validation = choose_windows(validation_ends, config.max_validation_windows, rng)
    # CPU chosen for reproducibility and modest local memory, not because it is the
    # only supported device. All parameters receive actual gradient updates.
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = WindowAutoencoder(
        len(config.channels), config.window, config.hidden, config.bottleneck
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    loss_fn = nn.SmoothL1Loss()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    start = time.monotonic()
    print(
        f"Training {len(training):,} windows; validation {len(validation):,}; parameters {sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch_start in range(0, len(training), config.batch_size):
            if batch_start == 0:
                rng.shuffle(training)
            ends = training[batch_start : batch_start + config.batch_size]
            batch = window_batch(values, ends, config.window, mean, scale)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(ends)
            seen += len(ends)
        model.eval()
        validation_total = 0
        with torch.inference_mode():
            for batch_start in range(0, len(validation), config.batch_size):
                ends = validation[batch_start : batch_start + config.batch_size]
                batch = window_batch(values, ends, config.window, mean, scale)
                validation_total += loss_fn(model(batch), batch).item() * len(ends)
        validation_loss = validation_total / len(validation)
        row = {
            "epoch": epoch,
            "train_loss": total / seen,
            "validation_loss": validation_loss,
            "elapsed_seconds": time.monotonic() - start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (reports / "training_history.json").write_text(json.dumps(history, indent=2))
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    checkpoint = {
        "state_dict": best_state,
        "config": config.to_dict(),
        "center": mean.tolist(),
        "scale": scale.tolist(),
        "source": "REAL MISSION DATA",
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "torch_version": str(torch.__version__),
    }
    torch.save(checkpoint, output / "autoencoder.pt")
    nominal_indices = np.flatnonzero(train_mask)
    nominal_indices = choose_windows(nominal_indices, 100000, rng)
    pca = PCA(n_components=2, svd_solver="full").fit(
        (values[nominal_indices] - mean) / scale
    )
    joblib.dump(pca, output / "pca_baseline.joblib")
    summary = {
        "source": "REAL MISSION DATA",
        "architecture": "384 → 64 GELU → 12 → 64 GELU → 384 temporal autoencoder",
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": "CPU",
        "training_nominal_rows": int(train_mask.sum()),
        "eligible_training_windows": len(train_ends),
        "sampled_training_windows": len(training),
        "sampled_validation_windows": len(validation),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss": best_loss,
        "config": config.to_dict(),
        "elapsed_seconds": time.monotonic() - start,
        "test_used_for_training_or_selection": False,
        "loss": "Smooth L1 reconstruction loss over each full trailing window",
        "inference_score": "Maximum squared standardized reconstruction residual across six channels at the window terminal sample",
        "training_filter": "Exclude all annotated anomalies, rare events and communication gaps; require complete windows within split",
        "preparation": preparation,
    }
    (reports / "training.json").write_text(json.dumps(summary, indent=2))
    (output / "model_card.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/esa_neural")
    parser.add_argument("--output", default="models/esa_neural")
    parser.add_argument("--reports", default="reports/esa_neural")
    args = parser.parse_args()
    train(Path(args.data), Path(args.output), Path(args.reports))
