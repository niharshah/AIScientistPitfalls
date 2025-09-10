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
    test_summary = {}
    # ---------- per-dataset plots ---------------------------------
    for ds_name, rec in experiment_data.items():
        epochs = rec.get("epochs", [])
        # ---- Figure 1: Macro-F1 curves ----
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            fig.suptitle(
                f"{ds_name} Macro-F1 over Epochs\nLeft: Train  Right: Validation",
                fontsize=14,
            )
            axes[0].plot(epochs, rec["metrics"]["train_macro_f1"], label="Train F1")
            axes[1].plot(epochs, rec["metrics"]["val_macro_f1"], label="Val F1")
            for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
                ax.set_title(ttl)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Macro-F1")
                ax.legend()
            fpath = os.path.join(working_dir, f"{ds_name}_macro_f1_curves.png")
            plt.savefig(fpath)
            plt.close()
        except Exception as e:
            print(f"Error creating F1 plot for {ds_name}: {e}")
            plt.close()

        # ---- Figure 2: Loss curves ----
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            fig.suptitle(
                f"{ds_name} Cross-Entropy Loss over Epochs\nLeft: Train  Right: Validation",
                fontsize=14,
            )
            axes[0].plot(epochs, rec["losses"]["train"], label="Train Loss")
            axes[1].plot(epochs, rec["losses"]["val"], label="Val Loss")
            for ax, ttl in zip(axes, ["Train Loss", "Validation Loss"]):
                ax.set_title(ttl)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
            fpath = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fpath)
            plt.close()
        except Exception as e:
            print(f"Error creating Loss plot for {ds_name}: {e}")
            plt.close()

        # collect test scores for comparison plot
        test_summary[ds_name] = rec.get("test_macro_f1", 0.0)

    # ---------- comparison bar chart ------------------------------
    try:
        fig = plt.figure(figsize=(8, 5))
        keys, vals = zip(*test_summary.items()) if test_summary else ([], [])
        plt.bar(keys, vals, color="skyblue")
        plt.title("Test Macro-F1 Comparison Across Datasets")
        plt.ylabel("Macro-F1")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fpath = os.path.join(working_dir, "all_datasets_test_macro_f1_bar.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating comparison bar chart: {e}")
        plt.close()

    # ---------- console summary -----------------------------------
    print("\nFinal Test Macro-F1 Scores:")
    for k, v in test_summary.items():
        print(f"{k:25s}: {v:.4f}")
