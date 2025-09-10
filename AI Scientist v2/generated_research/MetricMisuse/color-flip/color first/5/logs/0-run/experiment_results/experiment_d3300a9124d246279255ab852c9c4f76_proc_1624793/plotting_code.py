import matplotlib.pyplot as plt
import numpy as np
import os

# --- set up ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load experiment data ---
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    try:
        runs = experiment_data["weight_decay"]["SPR_BENCH"]
    except Exception as e:
        print(f"Error extracting runs: {e}")
        runs = {}

    # Pre-collect per-run arrays
    wd_vals, train_losses, val_losses, val_accs = [], [], [], []
    test_accs, test_cwa, test_swa, test_comp = [], [], [], []

    for wd, data in runs.items():
        wd_vals.append(float(wd))
        train_losses.append(data["losses"]["train"])
        val_losses.append(data["losses"]["val"])
        val_accs.append([m["acc"] for m in data["metrics"]["val"]])
        t = data["metrics"]["test"]
        test_accs.append(t["acc"])
        test_cwa.append(t["cwa"])
        test_swa.append(t["swa"])
        test_comp.append(t["compwa"])

    epochs = np.arange(1, len(train_losses[0]) + 1)

    # -------- Plot 1: loss curves --------
    try:
        plt.figure()
        for tl, vl, wd in zip(train_losses, val_losses, wd_vals):
            plt.plot(epochs, tl, label=f"train wd={wd}")
            plt.plot(epochs, vl, "--", label=f"val wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("SPR_BENCH: Train vs. Val Loss (all weight_decay)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_all_wd.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # -------- Plot 2: validation accuracy --------
    try:
        plt.figure()
        for va, wd in zip(val_accs, wd_vals):
            plt.plot(epochs, va, label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Accuracy over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_all_wd.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val accuracy plot: {e}")
        plt.close()

    # -------- Plot 3: test accuracy bar chart --------
    try:
        plt.figure()
        idx = np.arange(len(wd_vals))
        plt.bar(idx, test_accs, color="skyblue")
        plt.xticks(idx, wd_vals)
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Test Accuracy vs. weight_decay")
        plt.xlabel("weight_decay")
        fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar: {e}")
        plt.close()

    # -------- Plot 4: weighted accuracies bar chart --------
    try:
        plt.figure()
        width = 0.2
        idx = np.arange(len(wd_vals))
        plt.bar(idx - width, test_cwa, width, label="CWA")
        plt.bar(idx, test_swa, width, label="SWA")
        plt.bar(idx + width, test_comp, width, label="CompWA")
        plt.xticks(idx, wd_vals)
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH: Test Weighted Accuracies vs. weight_decay")
        plt.xlabel("weight_decay")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_test_weighted_accuracies_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating weighted acc bar: {e}")
        plt.close()
