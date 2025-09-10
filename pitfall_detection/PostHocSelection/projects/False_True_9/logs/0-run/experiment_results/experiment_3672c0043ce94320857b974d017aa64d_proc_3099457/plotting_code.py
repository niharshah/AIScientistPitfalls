import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is None:
    exit()

lr_logs = experiment_data.get("learning_rate", {})
lrs = sorted(lr_logs.keys(), key=float)

best_hwa = {}
# -------- plot 1: loss curves per lr -----------
for lr in lrs:
    log = lr_logs[lr]
    try:
        epochs_tr, tr_loss = zip(*log["losses"]["train"])
        epochs_val, val_loss = zip(*log["losses"]["val"])
        plt.figure()
        plt.plot(epochs_tr, tr_loss, label="Train Loss")
        plt.plot(epochs_val, val_loss, label="Validation Loss")
        plt.title(f"SPR_BENCH Loss Curves (lr={lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"SPR_BENCH_loss_curves_lr{lr}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for lr={lr}: {e}")
        plt.close()

# -------- plot 2: metric curves per lr ---------
for lr in lrs:
    log = lr_logs[lr]
    try:
        ep, swa, cwa, hwa = zip(*log["metrics"]["val"])
        best_hwa[lr] = max(hwa)
        plt.figure()
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, hwa, label="HWA")
        plt.title(f"SPR_BENCH Weighted Accuracies (lr={lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"SPR_BENCH_metric_curves_lr{lr}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for lr={lr}: {e}")
        plt.close()

# -------- plot 3: HWA comparison ---------------
try:
    plt.figure()
    for lr in lrs:
        ep, _, _, hwa = zip(*lr_logs[lr]["metrics"]["val"])
        plt.plot(ep, hwa, label=f"lr={lr}")
    plt.title("SPR_BENCH HWA Comparison Across Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    fname = "SPR_BENCH_HWA_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating HWA comparison plot: {e}")
    plt.close()

# -------- plot 4: Best HWA bar chart -----------
try:
    plt.figure()
    plt.bar(range(len(best_hwa)), [best_hwa[k] for k in lrs], tick_label=lrs)
    plt.title("SPR_BENCH Best HWA per Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Best HWA")
    fname = "SPR_BENCH_best_HWA_bar.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating best HWA bar chart: {e}")
    plt.close()

# -------- print summary ------------------------
for lr in lrs:
    print(f'Best HWA for lr={lr}: {best_hwa.get(lr, "N/A"):.4f}')
