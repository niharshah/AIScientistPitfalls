import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- Paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    bench = experiment_data["activation_function"]["SPR_BENCH"]
    activations = list(bench.keys())
    epochs = range(1, len(next(iter(bench.values()))["losses"]["train"]) + 1)

    # ---------- Figure 1: Loss curves ----------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for act in activations:
            axes[0].plot(epochs, bench[act]["losses"]["train"], label=act)
            axes[1].plot(epochs, bench[act]["losses"]["val"], label=act)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        for ax in axes:
            ax.legend()
        fig.suptitle(
            "SPR_BENCH Activation Function Sweep - Loss Curves\n"
            "Left: Training Loss, Right: Validation Loss"
        )
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve figure: {e}")
        plt.close()

    # ---------- Figure 2: Validation accuracy ----------
    try:
        plt.figure(figsize=(6, 4))
        for act in activations:
            val_acc = [m["acc"] for m in bench[act]["metrics"]["val"]]
            plt.plot(epochs, val_acc, marker="o", label=act)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("SPR_BENCH Activation Function Sweep - Validation Accuracy")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating validation accuracy figure: {e}")
        plt.close()

    # ---------- Figure 3: Test accuracy bar chart ----------
    try:
        test_accs = [bench[act]["metrics"]["test"]["acc"] for act in activations]
        plt.figure(figsize=(6, 4))
        plt.bar(activations, test_accs, color="skyblue")
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH Activation Function Sweep - Test Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR_BENCH_test_accuracy.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy figure: {e}")
        plt.close()

    # ---------- Print summary metrics ----------
    print("=== Test Accuracies ===")
    for act, acc in zip(activations, test_accs):
        print(f"{act:10s}: {acc:.3f}")
