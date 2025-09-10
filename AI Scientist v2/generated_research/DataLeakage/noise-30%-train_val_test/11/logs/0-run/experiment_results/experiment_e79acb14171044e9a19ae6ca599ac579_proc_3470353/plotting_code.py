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
    rec = experiment_data.get("hybrid_transformer", {})
    epochs = rec.get("epochs", [])
    train_f1 = rec.get("metrics", {}).get("train_macro_f1", [])
    val_f1 = rec.get("metrics", {}).get("val_macro_f1", [])
    train_loss = rec.get("losses", {}).get("train", [])
    val_loss = rec.get("losses", {}).get("val", [])
    test_f1 = rec.get("test_macro_f1", None)

    # -------- Figure 1 : Macro-F1 curves ---------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            "SPR_BENCH Macro-F1 over Epochs\nLeft: Train  Right: Validation",
            fontsize=14,
        )
        axes[0].plot(epochs, train_f1, label="Train Macro-F1", color="tab:blue")
        axes[1].plot(epochs, val_f1, label="Validation Macro-F1", color="tab:orange")
        for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro-F1")
            ax.set_ylim(0, 1)
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
        axes[0].plot(epochs, train_loss, label="Train Loss", color="tab:green")
        axes[1].plot(epochs, val_loss, label="Validation Loss", color="tab:red")
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
        fig = plt.figure(figsize=(5, 5))
        plt.bar(["hybrid_transformer"], [test_f1], color="skyblue")
        plt.title("SPR_BENCH Test Macro-F1")
        plt.ylabel("Macro-F1")
        plt.ylim(0, 1)
        fname = os.path.join(working_dir, "SPR_BENCH_test_macro_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Test Macro-F1 bar plot: {e}")
        plt.close()

    # -------- Console summary --------------------------------------------
    if test_f1 is not None:
        print(f"Final Test Macro-F1 : {test_f1:.4f}")
