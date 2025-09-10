import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# working directory for outputs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    models = experiment_data.keys()
    summary = {}

    # ---------- Figure 1 : Macro-F1 curves -------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            "SPR_BENCH Macro-F1 over Epochs\nLeft: Train   Right: Validation",
            fontsize=14,
        )
        for m in models:
            rec = experiment_data[m]
            epochs = rec["epochs"]
            axes[0].plot(epochs, rec["metrics"]["train"], label=m)
            axes[1].plot(epochs, rec["metrics"]["val"], label=m)
        for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro-F1")
            ax.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 plot: {e}")
        plt.close()

    # ---------- Figure 2 : Loss curves ----------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            "SPR_BENCH Cross-Entropy Loss over Epochs\nLeft: Train   Right: Validation",
            fontsize=14,
        )
        for m in models:
            rec = experiment_data[m]
            epochs = rec["epochs"]
            axes[0].plot(epochs, rec["losses"]["train"], label=m)
            axes[1].plot(epochs, rec["losses"]["val"], label=m)
        for ax, ttl in zip(axes, ["Train Loss", "Validation Loss"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # ---------- Figure 3 : Test Macro-F1 bar chart -----------------------
    try:
        keys, test_f1s = [], []
        for m in models:
            f1 = experiment_data[m].get("test_macro_f1", 0.0)
            keys.append(m)
            test_f1s.append(f1)
            summary[m] = f1
        plt.figure(figsize=(6, 5))
        plt.bar(keys, test_f1s, color="skyblue")
        plt.title("SPR_BENCH Test Macro-F1 by Model")
        plt.ylabel("Macro-F1")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_macro_f1_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Test Macro-F1 bar plot: {e}")
        plt.close()

    # ---------- Figure 4 & 5 : Confusion matrices -----------------------
    for m in models:
        try:
            preds = np.array(experiment_data[m]["predictions"])
            trues = np.array(experiment_data[m]["ground_truth"])
            cm = confusion_matrix(trues, preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(f"SPR_BENCH Confusion Matrix – {m}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(np.arange(cm.shape[1]))
            ax.set_yticks(np.arange(cm.shape[0]))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            fname = f"SPR_BENCH_confusion_matrix_{m}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {m}: {e}")
            plt.close()

    # ------------------ Console summary ---------------------------------
    print("\nFinal Test Macro-F1 Scores:")
    for k, v in summary.items():
        print(f"{k:15s}: {v:.4f}")
