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
    hs_runs = experiment_data["hidden_size_tuning"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    hs_runs = {}

# collect metrics
loss_tr, loss_val, acc_val = {}, {}, {}
for k, run in hs_runs.items():
    loss_tr[k] = run["losses"]["train"]
    loss_val[k] = run["losses"]["val"]
    acc_val[k] = [m["acc"] for m in run["metrics"]["val"]]

# ---------- plot loss curves ----------
try:
    plt.figure()
    for k in loss_tr:
        epochs = np.arange(1, len(loss_tr[k]) + 1)
        plt.plot(epochs, loss_tr[k], label=f"{k}-train")
        plt.plot(epochs, loss_val[k], "--", label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "SPR_BENCH – Loss Curves across Hidden Sizes\nLeft: Train, Right (dashed): Validation"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_hidden_sizes.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- plot accuracy curves ----------
try:
    plt.figure()
    for k in acc_val:
        epochs = np.arange(1, len(acc_val[k]) + 1)
        plt.plot(epochs, acc_val[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR_BENCH – Validation Accuracy across Hidden Sizes")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves_hidden_sizes.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ---------- bar chart of final accuracy ----------
try:
    plt.figure()
    hs_labels = list(acc_val.keys())
    final_accs = [vals[-1] for vals in acc_val.values()]
    plt.bar(hs_labels, final_accs, color="skyblue")
    plt.ylabel("Final Val Accuracy")
    plt.title("SPR_BENCH – Final Validation Accuracy by Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_final_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar plot: {e}")
    plt.close()

# ---------- print final accuracies ----------
for k, v in acc_val.items():
    print(f"{k}: final_val_acc={v[-1]:.3f}")
