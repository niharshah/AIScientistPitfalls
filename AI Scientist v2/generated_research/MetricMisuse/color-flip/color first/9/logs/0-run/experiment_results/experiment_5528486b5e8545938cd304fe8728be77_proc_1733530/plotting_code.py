import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data --------------
try:
    ed_all = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = ed_all.get("SPR_BENCH", None)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    # ---------- reshape ------------------
    tr_loss = ed["losses"]["train"]  # list[epoch] -> float
    val_loss = ed["losses"]["val"]  # list[epoch] -> float
    val_metrics = ed["metrics"]["val"]  # list[epoch] -> (cwa,swa,hwa,cna)
    test_metrics = ed["metrics"].get("test", None)  # (cwa,swa,hwa,cna)

    epochs = np.arange(1, len(tr_loss) + 1)
    stride = max(1, int(np.ceil(len(epochs) / 5)))
    sel_idx = list(range(0, len(epochs), stride))
    if sel_idx[-1] != len(epochs) - 1:
        sel_idx.append(len(epochs) - 1)

    # ---------- Plot 1: loss curves -------
    try:
        plt.figure()
        plt.plot(epochs[sel_idx], np.array(tr_loss)[sel_idx], "o-", label="Train")
        plt.plot(
            epochs[sel_idx], np.array(val_loss)[sel_idx], "x--", label="Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- Plot 2: Validation HWA ----
    try:
        plt.figure()
        hwa = [m[2] for m in val_metrics]
        plt.plot(epochs[sel_idx], np.array(hwa)[sel_idx], "s-", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_hwa.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ---------- Plot 3: CWA vs SWA --------
    try:
        plt.figure()
        cwa, swa = val_metrics[-1][0], val_metrics[-1][1]
        plt.scatter(cwa, swa, color="red")
        plt.text(cwa, swa, f"epoch {len(epochs)}")
        plt.xlabel("CWA")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH: Final Epoch CWA vs SWA Scatter")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa_swa_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        plt.close()

    # ---------- Plot 4: Test metrics bar --
    try:
        plt.figure()
        if test_metrics is None:
            test_metrics = val_metrics[-1]
        labels = ["CWA", "SWA", "HWA", "CNA"]
        plt.bar(labels, test_metrics, color=["blue", "orange", "green", "purple"])
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Test Metrics Summary\nDataset: SPR_BENCH")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar plot: {e}")
        plt.close()
