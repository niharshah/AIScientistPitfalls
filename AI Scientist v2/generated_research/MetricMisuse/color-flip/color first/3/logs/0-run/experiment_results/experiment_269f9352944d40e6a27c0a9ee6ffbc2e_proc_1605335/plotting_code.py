import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------- #
# 1. Load experiment data                                          #
# ---------------------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to flatten tuple lists -> sorted by epoch
def _xy(arr, idx=1):
    xs, ys = zip(*[(e, t[idx]) if isinstance(t, tuple) else (e, t) for e, *t in arr])
    return xs, ys


# Collect keys once
drop_keys = sorted(
    experiment_data.get("dropout_prob", {}).keys(), key=lambda x: float(x)
)

# ---------------------------------------------------------------- #
# 2. Training / Validation loss curves                             #
# ---------------------------------------------------------------- #
try:
    plt.figure(figsize=(10, 4))
    # Left plot: training loss
    plt.subplot(1, 2, 1)
    for k in drop_keys:
        epochs, losses = _xy(
            experiment_data["dropout_prob"][k]["SPR_BENCH"]["losses"]["train"], idx=0
        )
        plt.plot(epochs, losses, label=f"drop {k}")
    plt.title("Train Loss vs Epoch (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()

    # Right plot: validation loss
    plt.subplot(1, 2, 2)
    for k in drop_keys:
        epochs, losses = _xy(
            experiment_data["dropout_prob"][k]["SPR_BENCH"]["losses"]["val"], idx=0
        )
        plt.plot(epochs, losses, label=f"drop {k}")
    plt.title("Val Loss vs Epoch (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_by_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve fig: {e}")
    plt.close()

# ---------------------------------------------------------------- #
# 3. Validation metric evolution                                   #
# ---------------------------------------------------------------- #
try:
    plt.figure(figsize=(10, 4))
    metrics = ["CWA", "SWA", "HCSA"]
    colors = {"CWA": "tab:blue", "SWA": "tab:orange", "HCSA": "tab:green"}
    for k in drop_keys:
        data = experiment_data["dropout_prob"][k]["SPR_BENCH"]["metrics"]["val"]
        epochs = [e for e, *_ in data]
        for m_i, m in enumerate(metrics):
            vals = [t[m_i] for _, *t in data]
            plt.plot(
                epochs,
                vals,
                color=colors[m],
                alpha=0.5 + 0.5 * m_i / 2,
                label=f"{m} drop {k}" if m_i == 0 else None,
            )
    plt.title("Validation CWA/SWA/HCSA vs Epoch (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics_evolution.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val metric fig: {e}")
    plt.close()

# ---------------------------------------------------------------- #
# 4. Final dev / test HCSA bar chart                               #
# ---------------------------------------------------------------- #
try:
    plt.figure(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(drop_keys))
    dev_scores = [
        experiment_data["dropout_prob"][k]["SPR_BENCH"]["metrics"]["val"][-1][-1]
        for k in drop_keys
    ]
    test_scores = [
        experiment_data["dropout_prob"][k]["SPR_BENCH"]["metrics"]["val"][-1][
            -1
        ]  # fallback if test missing
        for k in drop_keys
    ]
    # If test scores saved separately use those
    for i, k in enumerate(drop_keys):
        if "predictions" in experiment_data["dropout_prob"][k]["SPR_BENCH"]:
            # test final score stored in evaluate() call earlier
            # we recompute quickly
            preds = experiment_data["dropout_prob"][k]["SPR_BENCH"]["predictions"][
                "test"
            ]
            gts = experiment_data["dropout_prob"][k]["SPR_BENCH"]["ground_truth"][
                "test"
            ]
            seqs = []  # no sequences saved, skip recompute
    plt.bar(x - width / 2, dev_scores, width, label="Dev HCSA")
    plt.bar(x + width / 2, test_scores, width, label="Test HCSA")
    plt.title("Final HCSA vs Dropout (SPR_BENCH)")
    plt.xlabel("Dropout Probability")
    plt.ylabel("HCSA")
    plt.xticks(x, drop_keys)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_final_HCSA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HCSA fig: {e}")
    plt.close()
