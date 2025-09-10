import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------- paths & loading ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("num_epochs", {}).get("SPR_BENCH", {})

# ------------------------- helper fetch ------------------------------
loss_tr = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])

acc_tr = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
acc_val = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]

test_metrics = ed.get("metrics", {}).get("test", {})
NRGS = ed.get("metrics", {}).get("NRGS")

preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# ------------------------------ plots --------------------------------
# 1) loss curves
try:
    if loss_tr and loss_val:
        plt.figure()
        plt.plot(loss_tr, label="Train")
        plt.plot(loss_val, label="Val")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
    else:
        print("Loss data unavailable; skipping loss plot.")
except Exception as e:
    print(f"Error creating loss plot: {e}")
finally:
    plt.close()

# 2) accuracy curves
try:
    if acc_tr and acc_val:
        plt.figure()
        plt.plot(acc_tr, label="Train")
        plt.plot(acc_val, label="Val")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
    else:
        print("Accuracy data unavailable; skipping accuracy plot.")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
finally:
    plt.close()

# 3) test metric bars
try:
    if test_metrics:
        metrics_names = ["ACC", "SWA", "CWA"]
        metrics_vals = [test_metrics.get(k.lower(), np.nan) for k in metrics_names]
        if NRGS is not None:
            metrics_names.append("NRGS")
            metrics_vals.append(NRGS)
        plt.figure()
        plt.bar(metrics_names, metrics_vals, color="steelblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test-Set Metrics\nBar heights reflect scores (0-1)")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
    else:
        print("Test metrics unavailable; skipping test metrics plot.")
except Exception as e:
    print(f"Error creating test metric plot: {e}")
finally:
    plt.close()

# 4) correct vs incorrect bar
try:
    if preds.size and gts.size:
        correct = int((preds == gts).sum())
        incorrect = int(preds.size - correct)
        plt.figure()
        plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"])
        plt.title("SPR_BENCH Prediction Outcomes\nCounts on Test Set")
        fname = os.path.join(working_dir, "SPR_BENCH_correct_incorrect.png")
        plt.savefig(fname)
    else:
        print("Prediction/GT arrays empty; skipping outcome plot.")
except Exception as e:
    print(f"Error creating outcome plot: {e}")
finally:
    plt.close()

# ---------------------------- printing -------------------------------
if test_metrics:
    print("=== SPR_BENCH TEST METRICS ===")
    for k, v in test_metrics.items():
        print(f"{k.upper():4}: {v:.3f}")
    if NRGS is not None:
        print(f"NRGS: {NRGS:.3f}")
