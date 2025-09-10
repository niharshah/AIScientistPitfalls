import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    edict = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    edict = {}

lr_dict = edict.get("learning_rate", {})  # structure: {lr_key: {"spr_bench": {...}}}
if not lr_dict:
    print("No learning_rate sweep found in experiment_data.")
    exit()


# Helper to grab series
def get_series(lr_key, field):
    d = lr_dict[lr_key]["spr_bench"][field]
    return d["train"], d["val"]


def get_metric_series(lr_key, metric_idx=2):  # 0:SWA 1:CWA 2:HWA
    mets = lr_dict[lr_key]["spr_bench"]["metrics"]["val"]
    return [m[metric_idx] for m in mets]


colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
lr_keys = list(lr_dict.keys())

# ----------------- Plot 1: loss curves -----------------
try:
    fig, ax = plt.subplots()
    for i, lr in enumerate(lr_keys):
        tr, val = get_series(lr, "losses")
        ax.plot(tr, "--", color=colors[i], label=f"train lr={lr}")
        ax.plot(val, "-", color=colors[i], label=f"val lr={lr}")
    ax.set_title("SPR_BENCH Loss Curves (Learning-Rate Sweep)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    fig.savefig(os.path.join(working_dir, "spr_loss_curves_lr_sweep.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------- Plot 2: validation HWA curves -----------------
try:
    fig, ax = plt.subplots()
    for i, lr in enumerate(lr_keys):
        hwa = get_metric_series(lr, 2)
        ax.plot(hwa, color=colors[i], label=f"val HWA lr={lr}")
    ax.set_title("SPR_BENCH Validation Harmonic-Weighted Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("HWA")
    ax.legend()
    fig.savefig(os.path.join(working_dir, "spr_val_hwa_curves_lr_sweep.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating val HWA curve plot: {e}")
    plt.close()

# ----------------- Plot 3: final test HWA bar -----------------
try:
    fig, ax = plt.subplots()
    test_hwa = [lr_dict[lr]["spr_bench"]["metrics"]["test"][2] for lr in lr_keys]
    ax.bar(lr_keys, test_hwa, color=colors[: len(lr_keys)])
    ax.set_title("SPR_BENCH Test Harmonic-Weighted Accuracy per Learning-Rate")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Test HWA")
    fig.savefig(os.path.join(working_dir, "spr_test_hwa_bar_lr_sweep.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating test HWA bar plot: {e}")
    plt.close()

# ----------------- Plot 4: SWA vs CWA scatter -----------------
try:
    fig, ax = plt.subplots()
    for i, lr in enumerate(lr_keys):
        swa, cwa, _ = lr_dict[lr]["spr_bench"]["metrics"]["test"]
        ax.scatter(swa, cwa, color=colors[i], label=f"lr={lr}")
        ax.annotate(lr, (swa, cwa))
    ax.set_title("SPR_BENCH Test SWA vs CWA")
    ax.set_xlabel("Shape-Weighted Accuracy (SWA)")
    ax.set_ylabel("Color-Weighted Accuracy (CWA)")
    ax.legend()
    fig.savefig(os.path.join(working_dir, "spr_test_swa_vs_cwa_scatter.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating SWA vs CWA scatter plot: {e}")
    plt.close()

# ----------------- Print summary -----------------
best_idx = np.argmax([lr_dict[lr]["spr_bench"]["metrics"]["test"][2] for lr in lr_keys])
best_lr = lr_keys[best_idx]
print("\n=== Test Metrics Summary ===")
for lr in lr_keys:
    swa, cwa, hwa = lr_dict[lr]["spr_bench"]["metrics"]["test"]
    print(f"lr={lr}: SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}")
print(f"\nBest learning rate based on HWA: {best_lr}")
