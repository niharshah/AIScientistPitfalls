import matplotlib.pyplot as plt
import numpy as np
import os

# I/O setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------- #
# Load stored experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------------------------------------------------- #
# Helper to compute simple accuracy
def simple_acc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


# -------------------------------------------------- #
# Iterate over each dataset and generate plots
for ds_name, ds in experiment_data.items():
    # Extract losses
    ep_tr, tr_losses = (
        zip(*ds["losses"]["train"]) if ds["losses"]["train"] else ([], [])
    )
    ep_val, val_losses = zip(*ds["losses"]["val"]) if ds["losses"]["val"] else ([], [])

    # Plot 1: Loss curves
    try:
        plt.figure()
        plt.plot(ep_tr, tr_losses, label="Train")
        plt.plot(ep_val, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} – Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # Extract SWA & CWA
    if ds["metrics"]["train"]:
        ep_m, tr_swa, tr_cwa = zip(*ds["metrics"]["train"])
    else:
        ep_m, tr_swa, tr_cwa = ([], [], [])
    if ds["metrics"]["val"]:
        _, val_swa, val_cwa = zip(*ds["metrics"]["val"])
    else:
        val_swa, val_cwa = ([], [])

    # Plot 2: SWA/CWA curves
    try:
        plt.figure()
        plt.plot(ep_m, tr_swa, label="Train SWA")
        plt.plot(ep_m, val_swa, label="Val SWA")
        plt.plot(ep_m, tr_cwa, "--", label="Train CWA")
        plt.plot(ep_m, val_cwa, "--", label="Val CWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(f"{ds_name} – SWA & CWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_weighted_acc.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA/CWA plot for {ds_name}: {e}")
        plt.close()

    # Extract ZRGS
    ep_z, zrg_scores = (
        zip(*ds["metrics"]["zrgs"]) if ds["metrics"]["zrgs"] else ([], [])
    )

    # Plot 3: Zero-shot Generalization Score
    try:
        plt.figure()
        plt.plot(ep_z, zrg_scores, color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("ZRGS")
        plt.title(f"{ds_name} – Zero-shot Generalization Score")
        fname = os.path.join(working_dir, f"{ds_name}_zrgs.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating ZRGS plot for {ds_name}: {e}")
        plt.close()

    # -------------------------------------------------- #
    # Print simple accuracies on self & hold-out tests
    acc_self = simple_acc(
        ds["ground_truth"]["self_test"], ds["predictions"]["self_test"]
    )
    acc_hold = simple_acc(
        ds["ground_truth"]["holdout_test"], ds["predictions"]["holdout_test"]
    )
    print(f"{ds_name}: Self-test Acc = {acc_self:.3f} | Hold-out Acc = {acc_hold:.3f}")
