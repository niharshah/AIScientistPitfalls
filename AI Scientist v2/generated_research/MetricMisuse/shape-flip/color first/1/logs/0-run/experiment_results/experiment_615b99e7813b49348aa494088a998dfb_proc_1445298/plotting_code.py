import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None and "SPR" in exp:
    lrs = [3e-4, 1e-3, 3e-3]  # learning rates used during training
    epochs = 6  # epochs per LR
    tr_loss = exp["SPR"]["losses"]["train"]
    val_loss = exp["SPR"]["losses"]["val"]
    val_metrics = exp["SPR"]["metrics"]["val"]

    # helper: slice the flat list into per-LR chunks
    def chunk(lst):
        return [lst[i * epochs : (i + 1) * epochs] for i in range(len(lrs))]

    tr_chunks, val_chunks = chunk(tr_loss), chunk(val_loss)
    acc_chunks = chunk([m["acc"] for m in val_metrics])
    cwa_chunks = chunk([m["cwa"] for m in val_metrics])
    swa_chunks = chunk([m["swa"] for m in val_metrics])
    hpa_chunks = chunk([m["hpa"] for m in val_metrics])

    # ---- Figure 1: Train Loss ----
    try:
        plt.figure(figsize=(5, 4), dpi=120)
        for lr, tl in zip(lrs, tr_chunks):
            plt.plot(range(1, epochs + 1), tl, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR Train Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_train_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating train-loss plot: {e}")
        plt.close()

    # ---- Figure 2: Validation Loss ----
    try:
        plt.figure(figsize=(5, 4), dpi=120)
        for lr, vl in zip(lrs, val_chunks):
            plt.plot(range(1, epochs + 1), vl, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating val-loss plot: {e}")
        plt.close()

    # ---- Figure 3: Validation Accuracy ----
    try:
        plt.figure(figsize=(5, 4), dpi=120)
        for lr, ac in zip(lrs, acc_chunks):
            plt.plot(range(1, epochs + 1), ac, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_acc.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating val-acc plot: {e}")
        plt.close()

    # ---- Figure 4: Color vs Shape Weighted Accuracy ----
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        for lr, cw, sw in zip(lrs, cwa_chunks, swa_chunks):
            ax[0].plot(range(1, epochs + 1), cw, label=f"lr={lr}")
            ax[1].plot(range(1, epochs + 1), sw, label=f"lr={lr}")
        ax[0].set_title("Left: Color-Weighted Acc")
        ax[1].set_title("Right: Shape-Weighted Acc")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Score")
            a.legend()
        fig.suptitle("SPR Color vs Shape Weighted Accuracies")
        fname = os.path.join(working_dir, "SPR_cwa_swa.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating CWA/SWA plot: {e}")
        plt.close()

    # ---- Print final HPA ----
    print("\nFinal Harmonic-Poly Accuracy per LR")
    for lr, hp in zip(lrs, [h[-1] for h in hpa_chunks]):
        print(f"  lr={lr:.0e}: HPA={hp:.3f}")
