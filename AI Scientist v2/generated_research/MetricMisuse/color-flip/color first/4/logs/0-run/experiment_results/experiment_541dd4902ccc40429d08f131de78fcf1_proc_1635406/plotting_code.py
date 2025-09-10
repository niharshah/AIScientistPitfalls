import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_data = {}


# ---------- helper ----------
def get_epochs(logs):
    return list(range(1, len(logs["losses"]["train"]) + 1))


# ---------- plot 1 : loss curves ----------
try:
    plt.figure()
    for dr, logs in exp_data.get("dropout_rate", {}).items():
        epochs = get_epochs(logs)
        plt.plot(
            epochs,
            logs["losses"]["train"],
            label=f"dropout {dr} - train",
            linestyle="--",
        )
        plt.plot(epochs, logs["losses"]["val"], label=f"dropout {dr} - val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Toy SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "sprbench_losses_vs_epoch.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2 : validation accuracy ----------
try:
    plt.figure()
    for dr, logs in exp_data.get("dropout_rate", {}).items():
        epochs = get_epochs(logs)
        acc = [m["acc"] for m in logs["metrics"]["val"]]
        plt.plot(epochs, acc, label=f"dropout {dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Toy SPR_BENCH: Validation Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "sprbench_val_accuracy.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- plot 3 : final weighted accuracies ----------
try:
    drs = list(exp_data.get("dropout_rate", {}).keys())
    metrics = ["cwa", "swa", "pcwa"]
    width = 0.2
    x = np.arange(len(drs))
    plt.figure()
    for i, m in enumerate(metrics):
        vals = [exp_data["dropout_rate"][dr]["metrics"]["val"][-1][m] for dr in drs]
        plt.bar(x + i * width, vals, width=width, label=m.upper())
    plt.xticks(x + width, drs)
    plt.ylabel("Score")
    plt.title("Toy SPR_BENCH: Final Weighted Accuracies by Dropout")
    plt.legend()
    fname = os.path.join(working_dir, "sprbench_final_weighted_accuracies.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()
