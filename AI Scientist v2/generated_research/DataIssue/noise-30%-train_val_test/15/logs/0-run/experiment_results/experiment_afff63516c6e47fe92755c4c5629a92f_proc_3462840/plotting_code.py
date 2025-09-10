import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    sweep = experiment_data.get("batch_size_sweep", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sweep = {}


# ---------------- helper to extract numeric bs --------
def _bs_key(k):
    try:
        return int(k.lstrip("bs"))
    except Exception:
        return k


# --------------- PLOT 1: loss curves ------------------
try:
    plt.figure()
    for bs_key, entry in sorted(sweep.items(), key=lambda kv: _bs_key(kv[0])):
        epochs = entry["epochs"]
        plt.plot(
            epochs,
            entry["losses"]["train"],
            label=f"train_bs{_bs_key(bs_key)}",
            linestyle="-",
        )
        plt.plot(
            epochs,
            entry["losses"]["val"],
            label=f"val_bs{_bs_key(bs_key)}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "SPR_BENCH Synthetic Data – Loss Curves\nLeft: Train, Right(dashed): Validation"
    )
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# --------------- PLOT 2: Macro-F1 curves -------------
try:
    plt.figure()
    for bs_key, entry in sorted(sweep.items(), key=lambda kv: _bs_key(kv[0])):
        epochs = entry["epochs"]
        plt.plot(
            epochs,
            entry["metrics"]["train"],
            label=f"train_bs{_bs_key(bs_key)}",
            linestyle="-",
        )
        plt.plot(
            epochs,
            entry["metrics"]["val"],
            label=f"val_bs{_bs_key(bs_key)}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(
        "SPR_BENCH Synthetic Data – Macro-F1 Curves\nLeft: Train, Right(dashed): Validation"
    )
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# --------------- PLOT 3: Test Macro-F1 bar chart -----
try:
    plt.figure()
    bs_list, scores = [], []
    for bs_key, entry in sorted(sweep.items(), key=lambda kv: _bs_key(kv[0])):
        bs_list.append(str(_bs_key(bs_key)))
        scores.append(entry.get("test_macroF1", np.nan))
    plt.bar(bs_list, scores, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Test Macro-F1")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Synthetic Data – Test Macro-F1 by Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test Macro-F1 bar plot: {e}")
    plt.close()

# ---------------- print evaluation metrics ------------
if sweep:
    print("\nFinal Test Macro-F1 scores:")
    for bs_key, entry in sorted(sweep.items(), key=lambda kv: _bs_key(kv[0])):
        print(
            f"  Batch size { _bs_key(bs_key):>3}:  {entry.get('test_macroF1', np.nan):.4f}"
        )
