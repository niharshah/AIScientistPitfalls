import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

branch = experiment_data.get("no_positional_encoding", {}).get("SPR_BENCH", {})
loss_tr = branch.get("losses", {}).get("train", [])
loss_val = branch.get("losses", {}).get("val", [])
metrics_val = branch.get("metrics", {}).get("val", [])
preds = branch.get("predictions", [])
gts = branch.get("ground_truth", [])
test_metrics = branch.get("metrics", {}).get("test", None)


# helper to unpack tuples safely
def unpack(list_of_tuples, idx, default=[]):
    try:
        return [t[idx] for t in list_of_tuples]
    except Exception:
        return default


epochs = unpack(loss_tr, 1)
tr_loss = unpack(loss_tr, 2)
val_loss = unpack(loss_val, 2)

m_epochs = unpack(metrics_val, 1)
cwa = unpack(metrics_val, 2)
swa = unpack(metrics_val, 3)
hwa = unpack(metrics_val, 4)
cna = unpack(metrics_val, 5)

# --------------- Plot 1: loss curves ---------------
try:
    if epochs and val_loss:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------------- Plot 2: validation metrics ---------------
try:
    if m_epochs:
        plt.figure()
        plt.plot(m_epochs, cwa, label="CWA")
        plt.plot(m_epochs, swa, label="SWA")
        plt.plot(m_epochs, hwa, label="HWA")
        plt.plot(m_epochs, cna, label="CNA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation Metrics Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_metrics.png"))
        plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# --------------- Plot 3: test correctness histogram ---------------
try:
    if preds and gts:
        correct = sum(int(p == t) for p, t in zip(preds, gts))
        incorrect = len(preds) - correct
        plt.figure()
        plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"])
        plt.ylabel("Count")
        plt.title("SPR_BENCH: Test Prediction Correctness")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_correctness.png"))
        plt.close()
except Exception as e:
    print(f"Error creating correctness plot: {e}")
    plt.close()

# ---------------- print final metrics ----------------
if test_metrics:
    lr, cwa_t, swa_t, hwa_t, cna_t = test_metrics
    print(
        f"Final Test Metrics | LR={lr:.3g} | CWA={cwa_t:.3f} | SWA={swa_t:.3f} | HWA={hwa_t:.3f} | CNA={cna_t:.3f}"
    )
