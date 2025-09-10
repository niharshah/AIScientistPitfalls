import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    epoch_runs = experiment_data["epochs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    epoch_runs = {}

# ------------- helper to find best run -------------
best_epochs, best_hwa = None, -1.0
for ep, d in epoch_runs.items():
    hwa = d["SPR_BENCH"]["test_hwa"]
    if hwa > best_hwa:
        best_hwa, best_epochs = int(ep), hwa

# ------------- Plot 1: test HWA vs #epochs ----------
try:
    plt.figure()
    xs, ys = [], []
    for ep, d in sorted(epoch_runs.items(), key=lambda x: int(x[0])):
        xs.append(int(ep))
        ys.append(d["SPR_BENCH"]["test_hwa"])
    plt.bar(xs, ys, color="skyblue")
    plt.xlabel("# Training Epochs")
    plt.ylabel("Test HWA")
    plt.title("SPR_BENCH: Test Harmonic Weighted Accuracy vs Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_HWA_vs_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# ------------- Plot 2: HWA curve (best run) ----------
try:
    plt.figure()
    best_metrics = epoch_runs[str(best_epochs)]["SPR_BENCH"]["metrics"]
    tr = best_metrics["train"]
    val = best_metrics["val"]
    ep_tr, hwa_tr = zip(*tr)
    ep_val, hwa_val = zip(*val)
    plt.plot(ep_tr, hwa_tr, label="Train HWA")
    plt.plot(ep_val, hwa_val, label="Val HWA")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title(f"SPR_BENCH (Best {best_epochs} Epochs): Train vs Val HWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, f"SPR_BENCH_train_val_HWA_best_{best_epochs}epochs.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# ------------- Plot 3: Loss curve (best run) ----------
try:
    plt.figure()
    best_losses = epoch_runs[str(best_epochs)]["SPR_BENCH"]["losses"]
    tr_l = best_losses["train"]
    val_l = best_losses["val"]
    ep_tr_l, loss_tr = zip(*tr_l)
    ep_val_l, loss_val = zip(*val_l)
    plt.plot(ep_tr_l, loss_tr, label="Train Loss")
    plt.plot(ep_val_l, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"SPR_BENCH (Best {best_epochs} Epochs): Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, f"SPR_BENCH_train_val_loss_best_{best_epochs}epochs.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

print(f"Best epoch setting: {best_epochs} epochs | Test HWA: {best_hwa:.4f}")
