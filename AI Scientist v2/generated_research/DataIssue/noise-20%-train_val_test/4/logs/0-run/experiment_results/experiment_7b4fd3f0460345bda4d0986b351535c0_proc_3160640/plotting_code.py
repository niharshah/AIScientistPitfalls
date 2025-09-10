import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
exp_path_try = [os.path.join(working_dir, "experiment_data.npy"), "experiment_data.npy"]
experiment_data = None
for p in exp_path_try:
    try:
        experiment_data = np.load(p, allow_pickle=True).item()
        break
    except Exception as e:
        print(f"Failed loading from {p}: {e}")
if experiment_data is None:
    raise FileNotFoundError("experiment_data.npy not found in expected paths")

spr_hist = experiment_data["batch_size"]["SPR_BENCH"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ---------- loss curves ----------
try:
    plt.figure()
    for i, (bs, hist) in enumerate(spr_hist.items()):
        epochs = hist["epochs"]
        plt.plot(
            epochs, hist["losses"]["train"], "--", color=colors[i], label=f"{bs}-train"
        )
        plt.plot(epochs, hist["losses"]["val"], "-", color=colors[i], label=f"{bs}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation (all batch sizes)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- F1 curves ----------
try:
    plt.figure()
    for i, (bs, hist) in enumerate(spr_hist.items()):
        epochs = hist["epochs"]
        plt.plot(
            epochs,
            hist["metrics"]["train_f1"],
            "--",
            color=colors[i],
            label=f"{bs}-train",
        )
        plt.plot(
            epochs, hist["metrics"]["val_f1"], "-", color=colors[i], label=f"{bs}-val"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(
        "SPR_BENCH Macro-F1 Curves\nLeft: Train, Right: Validation (all batch sizes)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ---------- bar chart of best val & test F1 ----------
try:
    plt.figure()
    bs_labels, best_vals, tests = [], [], []
    for bs, hist in spr_hist.items():
        bs_labels.append(bs.replace("bs_", ""))
        best_vals.append(hist["best_val_f1"])
        tests.append(hist["test_f1"])
    x = np.arange(len(bs_labels))
    width = 0.35
    plt.bar(x - width / 2, best_vals, width, label="Best Val F1")
    plt.bar(x + width / 2, tests, width, label="Test F1")
    plt.xticks(x, bs_labels)
    plt.ylim(0, 1)
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Validation vs Test F1 by Batch Size")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_test_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ---------- print summary ----------
print("Batch Size | Best Val F1 | Test F1")
for bs, hist in spr_hist.items():
    print(
        f"{bs.replace('bs_',''):>9} | {hist['best_val_f1']:.4f}     | {hist['test_f1']:.4f}"
    )
