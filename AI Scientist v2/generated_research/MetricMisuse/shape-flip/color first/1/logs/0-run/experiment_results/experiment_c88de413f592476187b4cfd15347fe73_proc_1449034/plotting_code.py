import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment logs ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_logs = experiment_data["edge_type_shuffled"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_logs = None

if spr_logs:
    epochs = np.array(spr_logs["epochs"])
    tr_losses = np.array(spr_logs["losses"]["train"])
    val_losses = np.array(spr_logs["losses"]["val"])
    val_metrics = spr_logs["metrics"]["val"]  # list of dicts
    hpa_vals = np.array([m["HPA"] for m in val_metrics])
    cwa_vals = np.array([m["CWA"] for m in val_metrics])
    swa_vals = np.array([m["SWA"] for m in val_metrics])

    # ---------- 1) Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title("SPR dataset — Training vs. Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2) HPA curve ----------
    try:
        plt.figure()
        plt.plot(epochs, hpa_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Poly Accuracy")
        plt.title("SPR dataset — Validation HPA over Epochs")
        fname = os.path.join(working_dir, "SPR_hpa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HPA curve: {e}")
        plt.close()

    # ---------- 3) CWA & SWA curves ----------
    try:
        plt.figure()
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, swa_vals, label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR dataset — Validation CWA vs. SWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_cwa_swa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA/SWA curve: {e}")
        plt.close()

    # ---------- 4) Test accuracy bar ----------
    try:
        preds = np.array(spr_logs["predictions"])
        gts = np.array(spr_logs["ground_truth"])
        test_acc = (preds == gts).mean() if len(preds) else 0.0
        plt.figure()
        plt.bar(["Test Accuracy"], [test_acc], color="tab:blue")
        plt.ylim(0, 1)
        plt.title("SPR dataset — Test Accuracy")
        fname = os.path.join(working_dir, "SPR_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
        print(f"Test Accuracy: {test_acc:.3f}")
    except Exception as e:
        print(f"Error creating test accuracy plot: {e}")
        plt.close()
