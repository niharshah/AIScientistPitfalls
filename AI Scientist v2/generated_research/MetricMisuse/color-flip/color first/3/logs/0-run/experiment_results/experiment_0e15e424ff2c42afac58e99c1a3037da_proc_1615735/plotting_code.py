import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("RandomClusterAssignment", {}).get("SPR_BENCH", {})
loss_train = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
metrics_val = exp.get("metrics", {}).get("val", [])


# Helper to split epoch/value pairs
def split_xy(pairs, idx=1):
    if not pairs:
        return [], []
    x, y = zip(*[(p[0], p[idx]) for p in pairs])
    return list(x), list(y)


# ---------- Plot 1: Loss curves ----------
try:
    e1, y1 = split_xy(loss_train)
    e2, y2 = split_xy(loss_val)
    plt.figure()
    if e1:
        plt.plot(e1, y1, label="Train")
    if e2:
        plt.plot(e2, y2, label="Validation")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()


# ---------- helper to fetch metric column ----------
def metric_curve(col):
    return split_xy(metrics_val, idx=col)


# columns: 1=CWA, 2=SWA, 3=HCSA, 4=SNWA
# ---------- Plot 2: HCSA ----------
try:
    e, hcs = metric_curve(3)
    plt.figure()
    if e:
        plt.plot(e, hcs, marker="o")
    plt.title("SPR_BENCH: Harmonic CSA (HCSA) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_HCSA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HCSA curve: {e}")
    plt.close()

# ---------- Plot 3: SNWA ----------
try:
    e, sn = metric_curve(4)
    plt.figure()
    if e:
        plt.plot(e, sn, marker="o", color="green")
    plt.title("SPR_BENCH: Sequence Novelty-Weighted Acc. (SNWA) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SNWA")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_SNWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SNWA curve: {e}")
    plt.close()

# ---------- Plot 4: CWA & SWA ----------
try:
    e, cwa = metric_curve(1)
    _, swa = metric_curve(2)
    plt.figure()
    if e:
        plt.plot(e, cwa, label="CWA")
    if e:
        plt.plot(e, swa, label="SWA")
    plt.title("SPR_BENCH: Color & Shape Weighted Acc. over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_CWA_SWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA/SWA curve: {e}")
    plt.close()

# ---------- print last-epoch key metrics ----------
if metrics_val:
    last = metrics_val[-1]
    print(
        f"Final Val Metrics -> Epoch {last[0]}: HCSA={last[3]:.3f}, SNWA={last[4]:.3f}"
    )
