import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data -----------
exp_path_candidates = [
    os.path.join(os.getcwd(), "experiment_data.npy"),
    os.path.join(working_dir, "experiment_data.npy"),
]
experiment_data = None
for p in exp_path_candidates:
    try:
        experiment_data = np.load(p, allow_pickle=True).item()
        break
    except Exception:
        continue
if experiment_data is None:
    raise FileNotFoundError("experiment_data.npy not found in expected locations.")

try:
    bs_dict = experiment_data["batch_size_tuning"]["SPR_BENCH"]
except KeyError as e:
    raise KeyError(f"Could not locate SPR_BENCH results in experiment_data: {e}")

# -------- gather data -----------
batch_sizes, train_f1, val_f1, train_loss, val_loss, best_dev_f1, test_f1 = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
for bs_key, content in sorted(bs_dict.items(), key=lambda x: int(x[0].split("_")[1])):
    bs = int(bs_key.split("_")[1])
    batch_sizes.append(bs)
    train_f1.append(content["metrics"]["train_f1"])
    val_f1.append(content["metrics"]["val_f1"])
    train_loss.append(content["losses"]["train"])
    val_loss.append(content["losses"]["val"])
    best_dev_f1.append(content.get("best_dev_f1", np.nan))
    test_f1.append(content.get("test_f1", np.nan))

epochs = range(1, 1 + max(len(v) for v in val_f1))


# -------- plotting helpers ----------
def safe_plot(fig_func, fname):
    try:
        fig_func()
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating {fname}: {e}")
    finally:
        plt.close()


# 1) Validation F1 curves
def plot_val_f1():
    plt.figure()
    for i, bs in enumerate(batch_sizes):
        plt.plot(val_f1[i], label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Validation Macro-F1 vs Epoch")
    plt.legend()


safe_plot(plot_val_f1, "SPR_BENCH_val_f1_curves.png")


# 2) Training F1 curves
def plot_train_f1():
    plt.figure()
    for i, bs in enumerate(batch_sizes):
        plt.plot(train_f1[i], label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training Macro-F1 vs Epoch")
    plt.legend()


safe_plot(plot_train_f1, "SPR_BENCH_train_f1_curves.png")


# 3) Validation loss curves
def plot_val_loss():
    plt.figure()
    for i, bs in enumerate(batch_sizes):
        plt.plot(val_loss[i], label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Validation Loss vs Epoch")
    plt.legend()


safe_plot(plot_val_loss, "SPR_BENCH_val_loss_curves.png")


# 4) Bar chart Best Dev F1
def plot_best_dev():
    plt.figure()
    plt.bar(range(len(batch_sizes)), best_dev_f1, tick_label=batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Best Dev Macro-F1")
    plt.title("SPR_BENCH: Best Dev Macro-F1 by Batch Size")


safe_plot(plot_best_dev, "SPR_BENCH_best_dev_f1_bar.png")


# 5) Bar chart Test F1
def plot_test_f1():
    plt.figure()
    plt.bar(range(len(batch_sizes)), test_f1, tick_label=batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Test Macro-F1")
    plt.title("SPR_BENCH: Test Macro-F1 by Batch Size")


safe_plot(plot_test_f1, "SPR_BENCH_test_f1_bar.png")

# -------- print summary -----------
print("BatchSize | BestDevF1 | TestF1")
for bs, dev, tst in zip(batch_sizes, best_dev_f1, test_f1):
    print(f"{bs:9d} | {dev:9.4f} | {tst:7.4f}")
