import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- LOAD DATA --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["binary_indicator"]["SPR_BENCH"]
    cfgs = ed["configs"]
    epochs = np.arange(1, len(ed["metrics"]["train_acc"][0]) + 1)

    # ---------------- ACCURACY CURVES ----------------
    try:
        plt.figure()
        for cfg, tr, va in zip(
            cfgs, ed["metrics"]["train_acc"], ed["metrics"]["val_acc"]
        ):
            plt.plot(epochs, tr, label=f"{cfg} train", linestyle="-")
            plt.plot(epochs, va, label=f"{cfg} val", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Training & Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------------- LOSS CURVES ----------------
    try:
        plt.figure()
        for cfg, tr, va in zip(cfgs, ed["losses"]["train"], ed["losses"]["val"]):
            plt.plot(epochs, tr, label=f"{cfg} train", linestyle="-")
            plt.plot(epochs, va, label=f"{cfg} val", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training & Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------------- RULE FIDELITY CURVES ----------------
    try:
        plt.figure()
        for cfg, rf in zip(cfgs, ed["metrics"]["rule_fidelity"]):
            plt.plot(epochs, rf, label=cfg)
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH: Rule Fidelity Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot: {e}")
        plt.close()

    # --------------- PRINT EVALUATION ----------------
    print("Best configuration stored:", ed.get("best_config", "N/A"))
