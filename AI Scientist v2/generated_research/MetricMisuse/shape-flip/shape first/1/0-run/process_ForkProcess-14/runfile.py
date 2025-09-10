import os
import numpy as np
import math

# ------------------------------------------------------------------
# locate and load data
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()


# ------------------------------------------------------------------
# helper to pick best value
def _best(values, minimize=False):
    """Return best (min or max) value from list while ignoring NaNs."""
    clean_vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not clean_vals:  # all NaNs or empty
        return float("nan")
    return min(clean_vals) if minimize else max(clean_vals)


# ------------------------------------------------------------------
# iterate over datasets and report
for dataset_name, data in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    # metrics -------------------------------------------------------
    metrics = data.get("metrics", {})
    if metrics:
        tr_acc = _best(metrics.get("train_acc", []), minimize=False)
        val_acc = _best(metrics.get("val_acc", []), minimize=False)
        val_swa = _best(metrics.get("val_SWA", []), minimize=False)
        ura = _best(metrics.get("URA", []), minimize=False)

        print(f"  train accuracy: {tr_acc:.4f}")
        print(f"  validation accuracy: {val_acc:.4f}")
        print(f"  validation SWA: {val_swa:.4f}")
        print(f"  URA: {ura:.4f}")

    # losses --------------------------------------------------------
    losses = data.get("losses", {})
    if losses:
        tr_loss = _best(losses.get("train", []), minimize=True)
        val_loss = _best(losses.get("val", []), minimize=True)

        print(f"  training loss: {tr_loss:.4f}")
        print(f"  validation loss: {val_loss:.4f}")
