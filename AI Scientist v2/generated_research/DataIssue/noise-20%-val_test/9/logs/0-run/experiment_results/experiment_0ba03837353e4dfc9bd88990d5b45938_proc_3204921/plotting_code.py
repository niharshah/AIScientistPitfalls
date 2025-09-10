import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data -------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------- helper to fetch arrays -------- #
def fetch(metric_key):
    out = {}
    for wd_str, run in experiment_data.get("weight_decay", {}).items():
        log = run.get("SPR_BENCH", {})
        out[float(wd_str)] = np.array(log.get(metric_key, []))
    return dict(sorted(out.items()))  # keep order


train_acc = fetch(("metrics", "train_acc"))
val_acc = fetch(("metrics", "val_acc"))
train_loss = fetch(("losses", "train"))
val_loss = fetch(("losses", "val"))
rba_curve = fetch(("metrics", "RBA"))

# final test metrics
test_acc = {}
test_rba = {}
for wd_str, run in experiment_data.get("weight_decay", {}).items():
    tm = run["SPR_BENCH"]["test_metrics"]
    test_acc[float(wd_str)] = tm["acc"]
    test_rba[float(wd_str)] = tm["RBA"]

# -------- plotting -------- #
plots_done = 0

try:
    plt.figure()
    for wd, vals in train_acc.items():
        plt.plot(vals, label=f"train wd={wd}")
        plt.plot(val_acc[wd], linestyle="--", label=f"val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves\nTraining vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
    plots_done += 1
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    plt.figure()
    for wd, vals in train_loss.items():
        plt.plot(vals, label=f"train wd={wd}")
        plt.plot(val_loss[wd], linestyle="--", label=f"val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nTraining vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    plots_done += 1
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    for wd, vals in rba_curve.items():
        plt.plot(vals, label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("RBA")
    plt.title("SPR_BENCH Rule-based Accuracy Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_RBA_curves.png")
    plt.savefig(fname)
    plt.close()
    plots_done += 1
except Exception as e:
    print(f"Error creating RBA plot: {e}")
    plt.close()

try:
    plt.figure()
    wds = list(sorted(test_acc))
    accs = [test_acc[wd] for wd in wds]
    plt.bar(range(len(wds)), accs, tick_label=wds)
    plt.xlabel("Weight Decay")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH Final Test Accuracy vs Weight Decay")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
    plots_done += 1
except Exception as e:
    print(f"Error creating test accuracy bar: {e}")
    plt.close()

try:
    plt.figure()
    rbas = [test_rba[wd] for wd in wds]
    plt.bar(range(len(wds)), rbas, tick_label=wds, color="orange")
    plt.xlabel("Weight Decay")
    plt.ylabel("Test RBA")
    plt.title("SPR_BENCH Final Test RBA vs Weight Decay")
    fname = os.path.join(working_dir, "SPR_BENCH_test_RBA_bar.png")
    plt.savefig(fname)
    plt.close()
    plots_done += 1
except Exception as e:
    print(f"Error creating test RBA bar: {e}")
    plt.close()

print(f"{plots_done} figures saved to {working_dir}")

# -------- print summary metrics -------- #
print("WeightDecay | TestAcc | TestRBA")
for wd in wds:
    print(f"{wd:>10} | {test_acc[wd]:.3f}  | {test_rba[wd]:.3f}")
