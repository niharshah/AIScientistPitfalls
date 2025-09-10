import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}


# Helper: safely fetch dict keys
def safe_get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


# Only proceed if data exists
spr_data = safe_get(exp, "num_epochs", "SPR_BENCH", default={})

# ------------------------------------------------------------------
# 1–4: loss & metric curves for each epoch option
for idx, (epoch_str, info) in enumerate(
    sorted(spr_data.items(), key=lambda x: int(x[0]))
):
    # limit to at most 5 overall figures; eight would violate guideline, so stop at 4*2 -> 8. We'll merge metric plots into one hence keep ≤5
    losses = info.get("losses", {})
    val_metrics = info.get("metrics", {}).get("val", [])
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    epochs = list(range(1, len(train_loss) + 1))

    # ----- plot A: loss curves -----
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.title(f"SPR_BENCH - Loss Curves (n_epochs={epoch_str})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"spr_bench_loss_curves_{epoch_str}epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot ({epoch_str}): {e}")
        plt.close()

# ------------------------------------------------------------------
# 5: Combined HWA curves for all configs (≤5 figures total)
try:
    plt.figure()
    for epoch_str, info in sorted(spr_data.items(), key=lambda x: int(x[0])):
        hwa_vals = [m["HWA"] for m in info.get("metrics", {}).get("val", [])]
        plt.plot(range(1, len(hwa_vals) + 1), hwa_vals, label=f"{epoch_str} epochs")
    plt.title("SPR_BENCH - Validation HWA across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic-Weighted-Accuracy")
    plt.legend()
    fname = "spr_bench_val_hwa_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating combined HWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print summary stats
for epoch_str, info in sorted(spr_data.items(), key=lambda x: int(x[0])):
    test_metrics = safe_get(info, "metrics", "test", default={})
    print(f"Epochs={epoch_str}: Test HWA={test_metrics.get('HWA', 'NA')}")
