import matplotlib.pyplot as plt
import numpy as np
import os

# working directory for outputs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    dropout_dict = experiment_data.get("dropout", {})
    # Collect final metrics for console printout
    summary = {}

    # -------- Figure 1 : Macro-F1 curves ---------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            "SPR_BENCH Macro-F1 over Epochs\nLeft: Train  Right: Validation",
            fontsize=14,
        )
        for key, rec in dropout_dict.items():
            epochs = rec["epochs"]
            axes[0].plot(epochs, rec["metrics"]["train_macro_f1"], label=key)
            axes[1].plot(epochs, rec["metrics"]["val_macro_f1"], label=key)
        for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro-F1")
            ax.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 plot: {e}")
        plt.close()

    # -------- Figure 2 : Loss curves -------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            "SPR_BENCH Cross-Entropy Loss over Epochs\nLeft: Train  Right: Validation",
            fontsize=14,
        )
        for key, rec in dropout_dict.items():
            epochs = rec["epochs"]
            axes[0].plot(epochs, rec["losses"]["train"], label=key)
            axes[1].plot(epochs, rec["losses"]["val"], label=key)
        for ax, ttl in zip(axes, ["Train Loss", "Validation Loss"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # -------- Figure 3 : Final Test Macro-F1 bar chart --------------------
    try:
        keys = []
        test_f1s = []
        for key, rec in dropout_dict.items():
            keys.append(key)
            test_f1s.append(rec.get("test_macro_f1", 0.0))
            summary[key] = rec.get("test_macro_f1", 0.0)
        fig = plt.figure(figsize=(8, 5))
        plt.bar(keys, test_f1s, color="skyblue")
        plt.title("SPR_BENCH Test Macro-F1 by Dropout Rate")
        plt.ylabel("Macro-F1")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_macro_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Test Macro-F1 bar plot: {e}")
        plt.close()

    # -------- Console summary --------------------------------------------
    print("\nFinal Test Macro-F1 Scores:")
    for k, v in summary.items():
        print(f"{k:20s} : {v:.4f}")
