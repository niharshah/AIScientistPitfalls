import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["batch_size_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

batch_sizes = exp["batch_sizes"]  # [32, 64, 128, 256, 512]
n_bs = len(batch_sizes)
epochs_per = 10
colors = plt.cm.viridis(np.linspace(0, 1, n_bs))


# helper: slice metric list into [n_bs, epochs_per]
def split_metric(lst):
    return [lst[i * epochs_per : (i + 1) * epochs_per] for i in range(n_bs)]


train_acc = split_metric(exp["metrics"]["train_acc"])
val_acc = split_metric(exp["metrics"]["val_acc"])
train_loss = split_metric(exp["losses"]["train"])
val_loss = split_metric(exp["losses"]["val"])
rule_fid = split_metric(exp["metrics"]["rule_fidelity"])

# 1) accuracy curves
try:
    plt.figure()
    for i, bs in enumerate(batch_sizes):
        epochs = np.arange(1, epochs_per + 1)
        plt.plot(
            epochs, train_acc[i], color=colors[i], linestyle="-", label=f"Train bs={bs}"
        )
        plt.plot(
            epochs, val_acc[i], color=colors[i], linestyle="--", label=f"Val bs={bs}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Training & Validation Accuracy per Epoch")
    plt.legend(fontsize=8, ncol=2)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) loss curves
try:
    plt.figure()
    for i, bs in enumerate(batch_sizes):
        epochs = np.arange(1, epochs_per + 1)
        plt.plot(
            epochs,
            train_loss[i],
            color=colors[i],
            linestyle="-",
            label=f"Train bs={bs}",
        )
        plt.plot(
            epochs, val_loss[i], color=colors[i], linestyle="--", label=f"Val bs={bs}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training & Validation Loss per Epoch")
    plt.legend(fontsize=8, ncol=2)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) rule fidelity curves
try:
    plt.figure()
    for i, bs in enumerate(batch_sizes):
        epochs = np.arange(1, epochs_per + 1)
        plt.plot(epochs, rule_fid[i], color=colors[i], marker="o", label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH Rule Fidelity per Epoch")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# 4) final test accuracy by batch size
try:
    gtruth = np.asarray(exp["ground_truth"])
    test_accs = []
    for preds in exp["predictions"]:  # len == n_bs
        acc = (preds == gtruth).mean()
        test_accs.append(acc)
    plt.figure()
    plt.bar(range(n_bs), test_accs, color=colors)
    plt.xticks(range(n_bs), batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH Final Test Accuracy vs Batch Size")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png"), dpi=150)
    plt.close()
    print({bs: round(acc, 3) for bs, acc in zip(batch_sizes, test_accs)})
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
