import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    logs = experiment_data["dropout_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    logs = {}

# ------------ pick best dropout ------------
best_dr, best_val = None, -1
for dr, log in logs.items():
    try:
        acc = log["metrics"]["val"][-1]["acc"]  # final-epoch val acc
        if acc > best_val:
            best_val, best_dr = acc, dr
    except Exception:
        continue

# ------------ plot 1: val acc vs epoch for all dropouts ------------
try:
    plt.figure()
    for dr, log in logs.items():
        accs = [m["acc"] for m in log["metrics"]["val"]]
        plt.plot(range(1, len(accs) + 1), accs, label=f"dropout={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epochs (SPR_BENCH)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_val_accuracy_vs_epoch_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val-acc plot: {e}")
    plt.close()

# ------------ plot 2: loss curves for best dropout ------------
try:
    best_log = logs[best_dr]
    tr_losses = best_log["losses"]["train"]
    val_losses = best_log["losses"]["val"]
    plt.figure()
    plt.plot(range(1, len(tr_losses) + 1), tr_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training/Validation Loss Curves (dropout={best_dr}, SPR_BENCH)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"spr_best_dropout_{best_dr}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------ plot 3: test metrics bar for best dropout ------------
try:
    metrics = best_log["metrics"]["test"]
    names = ["acc", "swa", "cwa", "nrgs"]
    vals = [metrics[k] for k in names]
    plt.figure()
    plt.bar(names, vals, color="skyblue")
    plt.ylim(0, 1)
    plt.title(f"Test Metrics (Best dropout={best_dr}, SPR_BENCH)")
    plt.tight_layout()
    fname = os.path.join(
        working_dir, f"spr_best_dropout_{best_dr}_test_metrics_bar.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics bar: {e}")
    plt.close()

# ------------ print evaluation summary ------------
if best_dr is not None:
    print(f"Best dropout rate: {best_dr}")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.3f}")
