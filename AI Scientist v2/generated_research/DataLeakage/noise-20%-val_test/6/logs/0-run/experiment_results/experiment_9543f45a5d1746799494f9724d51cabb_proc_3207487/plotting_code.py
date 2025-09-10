import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------- LOAD EXPERIMENT DATA ------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper: safely fetch data
def get_metric(bs, key1, key2):
    try:
        return experiment_data["batch_size"]["SPR_BENCH"][bs][key1][key2]
    except KeyError:
        return None


batch_sizes = sorted(
    experiment_data.get("batch_size", {}).get("SPR_BENCH", {}).keys(), key=int
)

# ------------------------- PLOT 1: ACCURACY ------------------------
try:
    plt.figure()
    for bs in batch_sizes:
        train_acc = get_metric(bs, "metrics", "train_acc")
        val_acc = get_metric(bs, "metrics", "val_acc")
        if train_acc is not None:
            epochs = np.arange(1, len(train_acc) + 1)
            plt.plot(epochs, train_acc, label=f"train bs={bs}")
        if val_acc is not None:
            epochs = np.arange(1, len(val_acc) + 1)
            plt.plot(epochs, val_acc, "--", label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training & Validation Accuracy vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------- PLOT 2: LOSS ----------------------------
try:
    plt.figure()
    for bs in batch_sizes:
        train_loss = get_metric(bs, "losses", "train")
        val_loss = get_metric(bs, "losses", "val")
        if train_loss is not None:
            epochs = np.arange(1, len(train_loss) + 1)
            plt.plot(epochs, train_loss, label=f"train bs={bs}")
        if val_loss is not None:
            epochs = np.arange(1, len(val_loss) + 1)
            plt.plot(epochs, val_loss, "--", label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training & Validation Loss vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- PLOT 3: RULE FIDELITY ------------------------
try:
    plt.figure()
    for bs in batch_sizes:
        rule_fid = get_metric(bs, "metrics", "rule_fidelity")
        if rule_fid is not None:
            epochs = np.arange(1, len(rule_fid) + 1)
            plt.plot(epochs, rule_fid, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH: Rule Fidelity vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# ------------- PLOT 4: FINAL TEST ACCURACY BAR CHART ---------------
try:
    test_accs = []
    for bs in batch_sizes:
        preds = experiment_data["batch_size"]["SPR_BENCH"][bs]["predictions"]
        gts = experiment_data["batch_size"]["SPR_BENCH"][bs]["ground_truth"]
        test_accs.append((preds == gts).mean())
    plt.figure()
    plt.bar(range(len(batch_sizes)), test_accs, tick_label=batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH: Final Test Accuracy by Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()
