import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment_data.npy: {e}")
    data = {}


# ---------------- helper to extract ----------------
def extract_curve(key, idx):
    """Return epochs and values for a given curve."""
    curve = data.get(key, {})
    train = np.array(curve.get("train", []))
    val = np.array(curve.get("val", []))
    if train.size and val.size:
        epochs = train[:, 0]
        return epochs, train[:, idx], val[:, idx]
    return np.array([]), np.array([]), np.array([])


# Loss curves
ep_loss, tr_loss, va_loss = extract_curve("losses", 1)
# Accuracy curves (idx=1 is un-weighted accuracy, 2=CWA, 3=SWA, 4=ComplexityWA)
ep_acc, tr_acc, va_acc = extract_curve("metrics", 1)
ep_cwa, tr_cwa, va_cwa = extract_curve("metrics", 2)
ep_swa, tr_swa, va_swa = extract_curve("metrics", 3)
ep_cpx, tr_cpx, va_cpx = extract_curve("metrics", 4)

curves = [
    ("Loss", (ep_loss, tr_loss, va_loss), "SPR_BENCH_loss_curve.png"),
    ("Accuracy", (ep_acc, tr_acc, va_acc), "SPR_BENCH_accuracy_curve.png"),
    ("Color-Weighted Accuracy", (ep_cwa, tr_cwa, va_cwa), "SPR_BENCH_CWA_curve.png"),
    ("Shape-Weighted Accuracy", (ep_swa, tr_swa, va_swa), "SPR_BENCH_SWA_curve.png"),
    (
        "Complexity-Weighted Accuracy",
        (ep_cpx, tr_cpx, va_cpx),
        "SPR_BENCH_ComplexityWA_curve.png",
    ),
]

# ---------------- plotting ----------------
for title, (ep, tr, va), fname in curves:
    try:
        if ep.size == 0:  # skip empty
            print(f"No data for {title}, skipping.")
            continue
        plt.figure()
        plt.plot(ep, tr, "o-", label="Train")
        plt.plot(ep, va, "s-", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(
            f"SPR_BENCH {title}\nDataset: Synthetic Primitive Reasoning Benchmark"
        )
        plt.legend()
        save_path = os.path.join(working_dir, fname)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating plot {title}: {e}")
        plt.close()

# ---------------- print final metrics ----------------
try:
    if ep_acc.size:
        print(f"Final Val Accuracy: {va_acc[-1]:.4f}")
    if ep_cwa.size:
        print(f"Final Val CWA: {va_cwa[-1]:.4f}")
    if ep_swa.size:
        print(f"Final Val SWA: {va_swa[-1]:.4f}")
    if ep_cpx.size:
        print(f"Final Val ComplexityWA: {va_cpx[-1]:.4f}")
except Exception as e:
    print(f"Error printing final metrics: {e}")
