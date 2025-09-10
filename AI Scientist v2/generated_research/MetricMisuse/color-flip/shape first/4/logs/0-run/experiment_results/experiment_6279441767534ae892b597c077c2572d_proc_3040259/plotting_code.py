import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def get_runs(model_key):
    runs = experiment_data.get(model_key, {}).get("SPR_BENCH", {})
    # sort by number of epochs (string keys -> int)
    return {int(k): v for k, v in runs.items()}


# ----- 1) aggregate loss curves for baseline -----
try:
    plt.figure(figsize=(6, 4))
    runs = get_runs("baseline")
    for epochs, rec in sorted(runs.items()):
        x = np.arange(1, len(rec["losses"]["train"]) + 1)
        plt.plot(x, rec["losses"]["train"], label=f"{epochs}ep-train")
        plt.plot(x, rec["losses"]["val"], "--", label=f"{epochs}ep-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Baseline LSTM (Packed)\nLeft: Train, Right: Val Loss Curves")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_baseline_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error plotting baseline loss curves: {e}")
    plt.close()

# ----- 2) aggregate loss curves for no-mask model -----
try:
    plt.figure(figsize=(6, 4))
    runs = get_runs("padding_mask_removal")
    for epochs, rec in sorted(runs.items()):
        x = np.arange(1, len(rec["losses"]["train"]) + 1)
        plt.plot(x, rec["losses"]["train"], label=f"{epochs}ep-train")
        plt.plot(x, rec["losses"]["val"], "--", label=f"{epochs}ep-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH LSTM (No Padding Mask)\nLeft: Train, Right: Val Loss Curves")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_nomask_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error plotting nomask loss curves: {e}")
    plt.close()

# ----- 3) final HWA comparison bar plot -----
try:
    plt.figure(figsize=(5, 3))
    epoch_vals = sorted(
        set(
            list(get_runs("baseline").keys())
            + list(get_runs("padding_mask_removal").keys())
        )
    )
    width = 0.35
    idx = np.arange(len(epoch_vals))
    hwa_base = [
        (
            get_runs("baseline").get(e, {"metrics": {"val": [np.nan]}})["metrics"][
                "val"
            ][-1]
            if e in get_runs("baseline")
            else np.nan
        )
        for e in epoch_vals
    ]
    hwa_nomask = [
        (
            get_runs("padding_mask_removal").get(e, {"metrics": {"val": [np.nan]}})[
                "metrics"
            ]["val"][-1]
            if e in get_runs("padding_mask_removal")
            else np.nan
        )
        for e in epoch_vals
    ]
    plt.bar(idx - width / 2, hwa_base, width, label="Baseline")
    plt.bar(idx + width / 2, hwa_nomask, width, label="NoMask")
    plt.xlabel("Training Epochs")
    plt.ylabel("Final HWA")
    plt.xticks(idx, epoch_vals)
    plt.title("SPR_BENCH Final Harmonic Weighted Accuracy\nBaseline vs No Padding Mask")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_final_HWA_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error plotting HWA comparison: {e}")
    plt.close()
