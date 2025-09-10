import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def safe_load(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return None


experiment_data = safe_load(os.path.join(working_dir, "experiment_data.npy"))
if experiment_data is None:
    exit()

spr_runs = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
if not spr_runs:
    print("No SPR_BENCH runs found in experiment_data.")
    exit()


def get_hmwa_list(metric_list):
    return [m.get("hmwa", 0.0) for m in metric_list]


global_best_lr, global_best_hmwa = None, -1
fig_count = 0
max_figs = 5

for lr_str, rec in spr_runs.items():
    epochs = list(range(1, len(rec["losses"]["train"]) + 1))
    train_loss = rec["losses"]["train"]
    val_loss = rec["losses"]["val"]
    hmwa = get_hmwa_list(rec["metrics"]["val"])
    cwa = [m.get("cwa", 0.0) for m in rec["metrics"]["val"]]
    swa = [m.get("swa", 0.0) for m in rec["metrics"]["val"]]
    best_hmwa_lr = max(hmwa) if hmwa else 0.0
    print(f"LR {lr_str}: best val HMWA = {best_hmwa_lr:.4f}")
    if best_hmwa_lr > global_best_hmwa:
        global_best_hmwa, global_best_lr = best_hmwa_lr, lr_str

    # -------- Plot 1: loss curves --------
    if fig_count < max_figs:
        try:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.title(f"SPR_BENCH Loss Curve (LR={lr_str})")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_BENCH_lr{lr_str}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
            fig_count += 1
        except Exception as e:
            print(f"Error creating loss plot for LR={lr_str}: {e}")
            plt.close()

    # -------- Plot 2: weighted accuracies --------
    if fig_count < max_figs:
        try:
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, hmwa, label="HMWA")
            plt.title(f"SPR_BENCH Weighted Accuracy Curve (LR={lr_str})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_BENCH_lr{lr_str}_metric_curve.png")
            plt.savefig(fname)
            plt.close()
            fig_count += 1
        except Exception as e:
            print(f"Error creating metric plot for LR={lr_str}: {e}")
            plt.close()

print(f"\nBest LR by validation HMWA: {global_best_lr} (HMWA={global_best_hmwa:.4f})")
