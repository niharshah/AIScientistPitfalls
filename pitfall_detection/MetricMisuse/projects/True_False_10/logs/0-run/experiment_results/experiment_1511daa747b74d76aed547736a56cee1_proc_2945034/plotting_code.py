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
    runs = experiment_data["gradient_clip_norm"]["SPR_BENCH"]
    clip_keys = list(runs.keys())  # e.g. ['none','0.1',...]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs, clip_keys = {}, []


# ---------------- helper ----------------
def get_epoch_counts(r):
    return range(1, len(r["losses"]["train"]) + 1)


# ---------------- Figure 1: loss curves ----------------
try:
    plt.figure()
    for ck in clip_keys:
        r = runs[ck]
        epochs = get_epoch_counts(r)
        plt.plot(epochs, r["losses"]["train"], "--", label=f"{ck}-train")
        plt.plot(epochs, r["losses"]["val"], "-", label=f"{ck}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss for Different Gradient Clip Norms")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_gradient_clip.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------- Figure 2: validation CRWA ----------------
try:
    plt.figure()
    for ck in clip_keys:
        r = runs[ck]
        epochs = get_epoch_counts(r)
        crwa_vals = [m["CRWA"] for m in r["metrics"]["val"]]
        plt.plot(epochs, crwa_vals, marker="o", label=ck)
    plt.xlabel("Epoch")
    plt.ylabel("CRWA")
    plt.title(
        "SPR_BENCH: Validation CRWA over Epochs\nLeft: Lower Clip, Right: Higher Clip"
    )
    plt.legend(title="clip_norm", fontsize=8)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_val_CRWA_gradient_clip.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CRWA plot: {e}")
    plt.close()

# ---------------- Figure 3: test CRWA bar chart ----------------
try:
    plt.figure()
    crwa_test = [runs[ck]["metrics"]["test"]["CRWA"] for ck in clip_keys]
    plt.bar(range(len(clip_keys)), crwa_test, tick_label=clip_keys)
    plt.ylabel("CRWA")
    plt.title("SPR_BENCH: Test CRWA by Gradient Clip Norm")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_CRWA_bar_gradient_clip.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test CRWA bar plot: {e}")
    plt.close()
