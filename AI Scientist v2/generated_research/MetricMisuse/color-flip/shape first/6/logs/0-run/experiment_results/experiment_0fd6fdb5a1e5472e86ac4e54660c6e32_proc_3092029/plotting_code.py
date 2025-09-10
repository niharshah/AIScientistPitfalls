import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data.get("EPOCH_PRE_tuning", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# helper to pick colors
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ---------- plot 1 : loss curves ----------
try:
    plt.figure()
    for idx, (run, data) in enumerate(runs.items()):
        tr, va = data["losses"]["train"], data["losses"]["val"]
        epochs = np.arange(1, len(tr) + 1)
        plt.plot(
            epochs,
            tr,
            marker="o",
            color=colors[idx % len(colors)],
            linestyle="-",
            label=f"{run}-train",
        )
        plt.plot(
            epochs,
            va,
            marker="x",
            color=colors[idx % len(colors)],
            linestyle="--",
            label=f"{run}-val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("EPOCH_PRE_tuning: Training vs. Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "EPOCH_PRE_tuning_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2 : SWA & CWA ----------
try:
    plt.figure()
    for idx, (run, data) in enumerate(runs.items()):
        swa, cwa = data["metrics"]["train"], data["metrics"]["val"]
        epochs = np.arange(1, len(swa) + 1)
        plt.plot(
            epochs,
            swa,
            marker="o",
            color=colors[idx % len(colors)],
            linestyle="-",
            label=f"{run}-SWA(train)",
        )
        plt.plot(
            epochs,
            cwa,
            marker="x",
            color=colors[idx % len(colors)],
            linestyle="--",
            label=f"{run}-CWA(val)",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EPOCH_PRE_tuning: Shape-Weighted vs. Color-Weighted Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "EPOCH_PRE_tuning_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- plot 3 : AIS ----------
try:
    plt.figure()
    for idx, (run, data) in enumerate(runs.items()):
        ais = data["AIS"]["val"]
        epochs = np.arange(1, len(ais) + 1)
        plt.plot(
            epochs, ais, marker="s", color=colors[idx % len(colors)], label=f"{run}-AIS"
        )
    plt.xlabel("Epoch")
    plt.ylabel("AIS")
    plt.title("EPOCH_PRE_tuning: Augmentation Invariance Score (val)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "EPOCH_PRE_tuning_AIS_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AIS plot: {e}")
    plt.close()


# ---------- confusion matrix for best run ----------
def confusion_matrix(gt, pr, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gt, pr):
        cm[g, p] += 1
    return cm


best_run, best_val = None, float("inf")
for run, data in runs.items():
    if data["losses"]["val"][-1] < best_val:
        best_val = data["losses"]["val"][-1]
        best_run = run

try:
    if best_run:
        data = runs[best_run]
        gt, pr = data["ground_truth"], data["predictions"]
        num_classes = max(max(gt), max(pr)) + 1
        cm = confusion_matrix(gt, pr, num_classes)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Confusion Matrix – {best_run}")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(os.path.join(working_dir, f"{best_run}_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print summary of best run ----------
if best_run:
    final_cwa = runs[best_run]["metrics"]["val"][-1]
    final_ais = runs[best_run]["AIS"]["val"][-1]
    print(
        f"Best run: {best_run} | Final Val Loss={best_val:.4f} | CWA={final_cwa:.3f} | AIS={final_ais:.3f}"
    )
