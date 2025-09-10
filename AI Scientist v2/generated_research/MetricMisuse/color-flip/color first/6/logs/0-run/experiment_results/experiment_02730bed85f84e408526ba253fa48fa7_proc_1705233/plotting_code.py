import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_key = "late_fusion_dual_lstm"
sub_key = "dual_channel"
if not experiment_data:
    exit()

loss_tr = experiment_data[exp_key][sub_key]["losses"]["train"]  # [(epoch, val)]
loss_va = experiment_data[exp_key][sub_key]["losses"]["val"]
met_va = experiment_data[exp_key][sub_key]["metrics"]["val"]  # [(epoch, dict)]

epochs_l = [e for e, _ in loss_tr]
tr_vals = [v for _, v in loss_tr]
va_vals = [v for _, v in loss_va]


def extract_metric(metric_name):
    return [d[metric_name] for _, d in met_va]


# ----------------------------------------------------------------------
# plot losses
try:
    plt.figure()
    plt.plot(epochs_l, tr_vals, label="Train")
    plt.plot(epochs_l, va_vals, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Late-Fusion Dual-LSTM on SPR-BENCH\nTraining vs Validation Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "sprbench_late_fusion_duallstm_loss_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ----------------------------------------------------------------------
# plot CWA, SWA, PCWA curves
for mname in ["CWA", "SWA", "PCWA"]:
    try:
        plt.figure()
        plt.plot(epochs_l, extract_metric(mname), marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(mname)
        plt.title(f"Late-Fusion Dual-LSTM on SPR-BENCH\nValidation {mname} over Epochs")
        plt.savefig(
            os.path.join(
                working_dir, f"sprbench_late_fusion_duallstm_{mname.lower()}_curve.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating {mname} plot: {e}")
        plt.close()

# ----------------------------------------------------------------------
# simple test-set evaluation
preds = experiment_data[exp_key][sub_key]["predictions"]
golds = experiment_data[exp_key][sub_key]["ground_truth"]
if preds and golds:
    acc = np.mean([p == g for p, g in zip(preds, golds)])
    best_pcwa = max([d["PCWA"] for _, d in met_va]) if met_va else float("nan")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Best validation PCWA: {best_pcwa:.4f}")
