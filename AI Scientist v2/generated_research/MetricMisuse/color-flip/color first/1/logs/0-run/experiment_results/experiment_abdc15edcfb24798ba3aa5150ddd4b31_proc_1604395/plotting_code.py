import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data.get("dropout_rate", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# ---------- prepare summary containers ----------
summary_drop, summary_cwa, summary_swa, summary_hmwa = [], [], [], []

# ---------- loss curves per dropout ----------
for run_name, run_dict in runs.items():
    losses = run_dict.get("losses", {})
    tr_losses = losses.get("train", [])
    val_losses = losses.get("val", [])
    epochs = range(1, len(tr_losses) + 1)
    try:
        plt.figure()
        plt.plot(epochs, tr_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{run_name}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{run_name}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {run_name}: {e}")
        plt.close()

    # collect summary metrics
    test_metrics = run_dict.get("metrics", {}).get("test", {})
    summary_drop.append(run_dict.get("dropout", np.nan))
    summary_cwa.append(test_metrics.get("cwa", np.nan))
    summary_swa.append(test_metrics.get("swa", np.nan))
    summary_hmwa.append(test_metrics.get("hmwa", np.nan))

# ---------- bar chart of test HMWA ----------
try:
    plt.figure()
    x = np.arange(len(summary_drop))
    plt.bar(x, summary_hmwa)
    plt.xticks(x, [f"drop={d:.1f}" for d in summary_drop])
    plt.ylabel("Test HMWA")
    plt.title("SPR_BENCH: Test HMWA vs Dropout Probability")
    fname = os.path.join(working_dir, "SPR_BENCH_test_HMWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HMWA bar chart: {e}")
    plt.close()

# ---------- print numeric summary ----------
print("Dropout | Test CWA | Test SWA | Test HMWA")
for d, c, s, h in zip(summary_drop, summary_cwa, summary_swa, summary_hmwa):
    print(f"{d:6.2f} | {c:8.3f} | {s:8.3f} | {h:9.3f}")
