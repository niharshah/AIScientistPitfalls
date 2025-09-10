import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR" in experiment_data:
    run = experiment_data["SPR"]
    epochs = run["epochs"]
    x = range(1, len(epochs) + 1)

    # helpers
    t_loss = run["losses"]["train"]
    v_loss = run["losses"]["val"]
    t_acc = [m["acc"] for m in run["metrics"]["train"]]
    v_acc = [m["acc"] for m in run["metrics"]["val"]]
    cwa = [m["cwa"] for m in run["metrics"]["val"]]
    swa = [m["swa"] for m in run["metrics"]["val"]]
    hpa = [m["hpa"] for m in run["metrics"]["val"]]

    # -------- Figure 1: Loss curves --------
    try:
        plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x, t_loss, label="Train")
        plt.plot(x, v_loss, label="Validation")
        plt.title("SPR – Left: Train Loss, Right: Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- Figure 2: Accuracy curves --------
    try:
        plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x, t_acc, label="Train")
        plt.plot(x, v_acc, label="Validation")
        plt.title("SPR – Left: Train Acc, Right: Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # -------- Figure 3: CWA & SWA --------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        ax[0].plot(x, cwa, label="CWA", color="tab:blue")
        ax[1].plot(x, swa, label="SWA", color="tab:orange")
        ax[0].set_title("Left: Color-Weighted Acc")
        ax[1].set_title("Right: Shape-Weighted Acc")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Score")
            a.legend()
        fig.suptitle("SPR Weighted Accuracies")
        fname = os.path.join(working_dir, "SPR_cwa_swa_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating CWA/SWA plot: {e}")
        plt.close()

    # -------- Figure 4: Harmonic Poly Accuracy --------
    try:
        plt.figure(figsize=(6, 4), dpi=120)
        plt.plot(x, hpa, label="HPA", color="tab:green")
        plt.title("SPR – Harmonic Poly Accuracy (HPA)")
        plt.xlabel("Epoch")
        plt.ylabel("HPA Score")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_hpa_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating HPA plot: {e}")
        plt.close()

    # -------- print summary metrics --------
    best_idx = int(np.argmax(v_acc))
    print(
        f"Best epoch @ {best_idx+1}: ValAcc={v_acc[best_idx]:.3f}, "
        f"CWA={cwa[best_idx]:.3f}, SWA={swa[best_idx]:.3f}, HPA={hpa[best_idx]:.3f}"
    )
else:
    print("No SPR data found.")
