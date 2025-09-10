import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------ setup & load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = experiment_data  # treat every top-level key as a dataset
summary = {}

# -------- Figure 1 : Macro-F1 curves -----------------------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle("Macro-F1 over Epochs\nLeft: Train  Right: Validation", fontsize=14)
    for name, rec in datasets.items():
        epochs = rec.get("epochs", [])
        axes[0].plot(epochs, rec["metrics"]["train_macro_f1"], label=name)
        axes[1].plot(epochs, rec["metrics"]["val_macro_f1"], label=name)
    for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
        ax.set_title(ttl)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.legend()
    plt.savefig(os.path.join(working_dir, "macro_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 plot: {e}")
    plt.close()

# -------- Figure 2 : Loss curves ---------------------------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle(
        "Cross-Entropy Loss over Epochs\nLeft: Train  Right: Validation", fontsize=14
    )
    for name, rec in datasets.items():
        epochs = rec.get("epochs", [])
        axes[0].plot(epochs, rec["losses"]["train"], label=name)
        axes[1].plot(epochs, rec["losses"]["val"], label=name)
    for ax, ttl in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_title(ttl)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Loss plot: {e}")
    plt.close()

# -------- Figure 3 : Final Test Macro-F1 bar chart ----------------------------
try:
    keys, test_f1s = [], []
    for name, rec in datasets.items():
        keys.append(name)
        test_f1s.append(rec.get("test_macro_f1", 0.0))
        summary[name] = rec.get("test_macro_f1", 0.0)
    fig = plt.figure(figsize=(8, 5))
    plt.bar(keys, test_f1s, color="skyblue")
    plt.title("Final Test Macro-F1 by Dataset")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "test_macro_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Test Macro-F1 bar plot: {e}")
    plt.close()

# -------- Figure 4 : Confusion Matrix (first 5 datasets) ----------------------
try:
    for idx, (name, rec) in enumerate(list(datasets.items())[:5]):
        preds = rec.get("predictions", [])
        trues = rec.get("ground_truth", [])
        if len(preds) == len(trues) and len(preds) > 0:
            cm = confusion_matrix(trues, preds, normalize="true")
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(f"{name} : Normalized Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
except Exception as e:
    print(f"Error creating Confusion Matrix: {e}")
    plt.close()

# -------- Console summary -----------------------------------------------------
print("\nFinal Test Macro-F1 Scores:")
for k, v in summary.items():
    print(f"{k:25s}: {v:.4f}")
