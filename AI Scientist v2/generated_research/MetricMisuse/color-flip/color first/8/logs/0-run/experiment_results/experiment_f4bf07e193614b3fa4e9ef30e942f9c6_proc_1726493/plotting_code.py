import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

# -------- Plot 1: losses ------------------------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    for exp_key, exp_dict in experiment_data["num_epochs"].items():
        # training losses
        train = exp_dict["losses"]["train"]
        if train:
            epochs, losses = zip(*train)
            plt.plot(epochs, losses, "--", label=f"train-{exp_key}")
        # validation losses
        val = exp_dict["losses"]["val"]
        if val:
            epochs, losses = zip(*val)
            plt.plot(epochs, losses, "-", label=f"val-{exp_key}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Synthetic SPR: Training and Validation Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "spr_loss_curves_max_epochs_50_75_100.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- Plot 2: validation CSHM --------------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    for exp_key, exp_dict in experiment_data["num_epochs"].items():
        val_metrics = exp_dict["metrics"]["val"]  # list of (epoch, cwa, swa, cshm)
        if val_metrics:
            epochs = [t[0] for t in val_metrics]
            cshm = [t[3] for t in val_metrics]
            plt.plot(epochs, cshm, label=f"CSHM-{exp_key}")
    plt.xlabel("Epoch")
    plt.ylabel("CSHM")
    plt.title("Synthetic SPR: Validation Colour-Shape Harmonic Mean (CSHM)")
    plt.legend()
    fname = os.path.join(
        working_dir, "spr_validation_cshm_curves_max_epochs_50_75_100.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CSHM plot: {e}")
    plt.close()
