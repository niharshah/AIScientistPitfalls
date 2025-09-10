import os
import numpy as np

# ---------------------------------------------------------------------
# 0. resolve working directory and load stored experiment data
# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
exp_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(exp_path, allow_pickle=True).item()

# ---------------------------------------------------------------------
# 1. iterate over datasets and print final / best metrics
# ---------------------------------------------------------------------
for dataset_name, data in experiment_data.items():
    print(f"\n{dataset_name}")  # dataset header

    # ---------- training ----------
    if data.get("losses", {}).get("train"):
        train_loss_final = data["losses"]["train"][-1]
        print(f"train loss: {train_loss_final:.6f}")

    if data.get("metrics", {}).get("train"):
        train_final = data["metrics"]["train"][-1]
        for k, v in train_final.items():
            print(f"train {k}: {v:.6f}")

    # ---------- validation ----------
    if data.get("losses", {}).get("val"):
        val_loss_final = data["losses"]["val"][-1]
        print(f"validation loss: {val_loss_final:.6f}")

    if data.get("metrics", {}).get("val"):
        val_final = data["metrics"]["val"][-1]
        for k, v in val_final.items():
            print(f"validation {k}: {v:.6f}")

    # ---------- test ----------
    if "test_metrics" in data:
        for k, v in data["test_metrics"].items():
            if k == "loss":
                print(f"test loss: {v:.6f}")
            else:
                print(f"test {k}: {v:.6f}")
