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
    experiment_data = {}


def nested_get(d, keys, default=[]):
    for k in keys:
        d = d.get(k, {})
    return d if isinstance(d, list) else default


final_ccwa = {}

# -------- per-dataset visualizations (max 2 each) ----------
for ds_name, logs in experiment_data.items():
    train_loss = nested_get(logs, ["losses", "train"])
    val_loss = nested_get(logs, ["losses", "val"])
    val_ccwa = nested_get(logs, ["metrics", "val_CCWA"])
    epochs = list(range(1, max(len(train_loss), len(val_loss)) + 1))

    # Plot 1: Loss curves
    try:
        plt.figure()
        if train_loss:
            plt.plot(epochs[: len(train_loss)], train_loss, label="Train Loss")
        if val_loss:
            plt.plot(epochs[: len(val_loss)], val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} – Train vs. Val Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # Plot 2: Validation CCWA curve
    try:
        if val_ccwa:
            plt.figure()
            plt.plot(range(1, len(val_ccwa) + 1), val_ccwa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("CCWA")
            plt.title(f"{ds_name} – Validation CCWA")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_val_CCWA.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating CCWA plot for {ds_name}: {e}")
        plt.close()

    if val_ccwa:
        final_ccwa[ds_name] = val_ccwa[-1]

# -------- summary bar chart (max one extra figure) ----------
try:
    if final_ccwa:
        plt.figure()
        names, scores = zip(*final_ccwa.items())
        plt.bar(names, scores)
        for i, s in enumerate(scores):
            plt.text(i, s, f"{s:.3f}", ha="center", va="bottom")
        plt.ylabel("Final Epoch CCWA")
        plt.title("Final Validation CCWA by Dataset")
        plt.savefig(os.path.join(working_dir, "final_CCWA_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating summary CCWA plot: {e}")
    plt.close()

# -------- print numerical summary ----------
print("Final Validation CCWA:")
for ds, score in final_ccwa.items():
    print(f"  {ds}: {score:.4f}")
