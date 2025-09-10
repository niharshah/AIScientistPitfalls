import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["SPR_DeepRGCN"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

if run:
    epochs = run["epochs"]
    tr_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    tr_cmp = run["metrics"]["train"]["CmpWA"]
    val_cmp = run["metrics"]["val"]["CmpWA"]
    test_metrics = run.get("test_metrics", {})

    # ---------- 1) loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, "--", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR synthetic – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2) CmpWA curves ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_cmp, label="Train")
        plt.plot(epochs, val_cmp, "--", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR synthetic – Training vs Validation CmpWA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_CmpWA_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating CmpWA plot: {e}")
        plt.close()

    # ---------- 3) test metrics summary ----------
    try:
        metrics_names = ["CmpWA", "CWA", "SWA"]
        values = [test_metrics.get(k, 0.0) for k in metrics_names]
        plt.figure()
        bars = plt.bar(metrics_names, values, color="skyblue")
        for bar, val, name in zip(bars, values, metrics_names):
            txt = f"{val:.2f}"
            if name == "CmpWA":
                txt += f'\nloss={test_metrics.get("loss", 0):.2f}'
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR synthetic – Test-set Weighted Accuracies")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_test_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()
