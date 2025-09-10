import os
import numpy as np

# --------------------------------------------------
# locate and load the saved experiment artefacts
# --------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()


# --------------------------------------------------
# helper to format floating numbers uniformly
# --------------------------------------------------
def fmt(x):
    return f"{x:.3f}" if isinstance(x, (float, int)) else str(x)


# --------------------------------------------------
# iterate over each dataset and print final metrics
# --------------------------------------------------
for dataset_name, data in experiment_data.items():
    print(f"\n{dataset_name}")  # dataset header

    # ---------- training ----------
    train_losses = data.get("losses", {}).get("train", [])
    if train_losses:
        print("training loss:", fmt(train_losses[-1]))

    # ---------- validation ----------
    val_losses = data.get("losses", {}).get("val", [])
    if val_losses:
        print("validation loss:", fmt(val_losses[-1]))

    val_metrics = data.get("metrics", {}).get("val", [])
    if val_metrics:
        last_val = val_metrics[-1]
        if "acc" in last_val:
            print("validation accuracy:", fmt(last_val["acc"]))
        if "swa" in last_val:
            print("validation shape weighted accuracy:", fmt(last_val["swa"]))
        if "cwa" in last_val:
            print("validation color weighted accuracy:", fmt(last_val["cwa"]))
        if "nrgs" in last_val:
            print("validation NRGS:", fmt(last_val["nrgs"]))

    # ---------- test ----------
    test_metrics = data.get("metrics", {}).get("test", {})
    if test_metrics:
        if "loss" in test_metrics:
            print("test loss:", fmt(test_metrics["loss"]))
        if "acc" in test_metrics:
            print("test accuracy:", fmt(test_metrics["acc"]))
        if "swa" in test_metrics:
            print("test shape weighted accuracy:", fmt(test_metrics["swa"]))
        if "cwa" in test_metrics:
            print("test color weighted accuracy:", fmt(test_metrics["cwa"]))
        if "nrgs" in test_metrics:
            print("test NRGS:", fmt(test_metrics["nrgs"]))
