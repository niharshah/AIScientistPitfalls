import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load experiment data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    e = experiment_data["dropout_rate"]["spr_bench"]
except Exception as exc:
    print(f"Error loading experiment data: {exc}")
    exit()

drops = e["rates"]
train_loss = e["losses"]["train"]  # list[ list[epoch_loss] ]
val_loss = e["losses"]["val"]
train_met = e["metrics"]["train"]  # list[ list[(swa,cwa,hwa)] ]
val_met = e["metrics"]["val"]
test_met = e["metrics"]["test"]  # list[(swa,cwa,hwa)]


# helper to pull a metric across epochs
def metric_series(metric_list, idx):
    return [[epoch[idx] for epoch in dr] for dr in metric_list]


swa_tr = metric_series(train_met, 0)
cwa_tr = metric_series(train_met, 1)
hwa_test = [m[2] for m in test_met]
epochs = range(1, len(train_loss[0]) + 1)

# ------------- figure 1: loss curves -------------
try:
    plt.figure(figsize=(6, 4))
    for d, tr, vl in zip(drops, train_loss, val_loss):
        plt.plot(epochs, tr, "--", label=f"train d={d}")
        plt.plot(epochs, vl, "-", label=f"val d={d}")
    plt.title("SPR_BENCH Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(ncol=2)
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as exc:
    print(f"Error creating loss plot: {exc}")
    plt.close()

# ------------- figure 2: SWA curves -------------
try:
    plt.figure(figsize=(6, 4))
    for d, met in zip(drops, swa_tr):
        plt.plot(epochs, met, label=f"dropout={d}")
    plt.title("SPR_BENCH Shape-Weighted Accuracy (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as exc:
    print(f"Error creating SWA plot: {exc}")
    plt.close()

# ------------- figure 3: CWA curves -------------
try:
    plt.figure(figsize=(6, 4))
    for d, met in zip(drops, cwa_tr):
        plt.plot(epochs, met, label=f"dropout={d}")
    plt.title("SPR_BENCH Color-Weighted Accuracy (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_cwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as exc:
    print(f"Error creating CWA plot: {exc}")
    plt.close()

# ------------- figure 4: test HWA bar chart -------------
try:
    plt.figure(figsize=(5, 4))
    plt.bar([str(d) for d in drops], hwa_test, color="steelblue")
    plt.title("SPR_BENCH Test HWA vs Dropout")
    plt.xlabel("Dropout Rate")
    plt.ylabel("HWA")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_test_hwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as exc:
    print(f"Error creating test HWA plot: {exc}")
    plt.close()

# ------------- print summary table -------------
print("Test HWA by dropout:")
for d, h in zip(drops, hwa_test):
    print(f"  dropout={d:>3}: HWA={h:.4f}")
