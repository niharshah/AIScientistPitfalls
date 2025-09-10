import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    exp = experiment_data["SPR_BENCH"]
    loss_tr = np.array(exp["losses"]["train"])
    loss_val = np.array(exp["losses"]["val"])
    f1_tr = np.array(exp["metrics"]["train_f1"])
    f1_val = np.array(exp["metrics"]["val_f1"])
    rea_dev = np.array(exp["metrics"]["REA_dev"])
    epochs = np.arange(1, len(loss_tr) + 1)

    # ---------- 1) Train / Val loss ----------
    try:
        plt.figure()
        plt.plot(epochs, loss_tr, label="Train")
        plt.plot(epochs, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2) Train / Val macro-F1 ----------
    try:
        plt.figure()
        plt.plot(epochs, f1_tr, label="Train F1")
        plt.plot(epochs, f1_val, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Train vs Validation Macro-F1")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # ---------- 3) Rule-Extraction Accuracy on Dev ----------
    try:
        plt.figure()
        plt.plot(epochs, rea_dev, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Extraction Accuracy")
        plt.title("SPR_BENCH: REA on Dev Set")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_REA_dev.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating REA plot: {e}")
        plt.close()

    # ---------- Simple metric printout ----------
    if len(f1_val) > 0:
        best_idx = int(f1_val.argmax())
        print(f"Best Validation Macro-F1: {f1_val[best_idx]:.4f} at epoch {best_idx+1}")
