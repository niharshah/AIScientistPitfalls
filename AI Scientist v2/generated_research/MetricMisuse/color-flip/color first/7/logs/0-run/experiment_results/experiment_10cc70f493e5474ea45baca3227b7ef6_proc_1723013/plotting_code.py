import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_key = "SPR_BENCH"
if data_key not in experiment_data:
    print(f"{data_key} not found in experiment data.")
    exit()

loss_train = experiment_data[data_key]["losses"]["train"]
loss_val = experiment_data[data_key]["losses"]["val"]
acc_train = experiment_data[data_key]["metrics"]["train_acc"]
acc_val = experiment_data[data_key]["metrics"]["val_acc"]
cpx_train = experiment_data[data_key]["metrics"]["train_cpx"]
cpx_val = experiment_data[data_key]["metrics"]["val_cpx"]


# ---------- helper for complexity weighted accuracy ----------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def complexity_weight(seq):
    return count_color_variety(seq) * count_shape_variety(seq)


def cpx_weighted_accuracy(seqs, y_true, y_pred):
    w = np.array([complexity_weight(s) for s in seqs])
    correct = (y_true == y_pred).astype(int)
    return (w * correct).sum() / (w.sum() + 1e-9)


# ---------- PLOTS ----------
try:
    plt.figure()
    plt.plot(loss_train, label="Train")
    plt.plot(loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(acc_train, label="Train")
    plt.plot(acc_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves\nTrain vs Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(cpx_train, label="Train")
    plt.plot(cpx_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH Cpx-Weighted Accuracy Curves\nTrain vs Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_cpx_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cpx-weighted accuracy plot: {e}")
    plt.close()

# ---------- final test metrics ----------
try:
    preds = np.array(experiment_data[data_key]["predictions"])
    gtruth = np.array(experiment_data[data_key]["ground_truth"])
    seqs_test = (
        experiment_data.get(data_key).get("seq_test")
        if "seq_test" in experiment_data.get(data_key, {})
        else None
    )
    plain_acc = (preds == gtruth).mean()
    if seqs_test is None:
        # seqs not saved; gracefully skip cpx metric
        print(f"TEST PlainAcc={plain_acc:.3f}")
    else:
        cpx_acc = cpx_weighted_accuracy(seqs_test, gtruth, preds)
        print(f"TEST PlainAcc={plain_acc:.3f} | CpxWA={cpx_acc:.3f}")
except Exception as e:
    print(f"Error computing test metrics: {e}")
