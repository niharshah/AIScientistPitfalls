import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------- load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_names = list(experiment_data.keys())
dataset = "SPR_BENCH"


# -------------------------------------------------- helper to pull curves
def extract_curve(exp, split, key):
    records = experiment_data[exp][dataset][key][split]
    epochs = [d["epoch"] for d in records]
    values = [d["loss" if key == "losses" else "macro_f1"] for d in records]
    return epochs, values


# -------------------------------------------------- 1) combined loss curves
try:
    plt.figure()
    for exp in exp_names:
        ep, tr = extract_curve(exp, "train", "losses")
        _, va = extract_curve(exp, "val", "losses")
        plt.plot(ep, tr, marker="o", label=f"{exp}-train")
        plt.plot(ep, va, marker="x", linestyle="--", label=f"{exp}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves (Train vs. Val)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_LossCurves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss Curves: {e}")
    plt.close()

# -------------------------------------------------- 2) combined macro-F1 curves
try:
    plt.figure()
    for exp in exp_names:
        ep, tr = extract_curve(exp, "train", "metrics")
        _, va = extract_curve(exp, "val", "metrics")
        plt.plot(ep, tr, marker="o", label=f"{exp}-train")
        plt.plot(ep, va, marker="x", linestyle="--", label=f"{exp}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves (Train vs. Val)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_F1Curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 Curves: {e}")
    plt.close()

# -------------------------------------------------- 3-5) bar charts GT vs Pred per experiment
for exp in exp_names:
    try:
        rec = experiment_data[exp][dataset]
        gt = np.array(rec["ground_truth"])
        pr = np.array(rec["predictions"])
        labels = sorted(set(gt))
        counts_gt = [np.sum(gt == l) for l in labels]
        counts_pr = [np.sum(pr == l) for l in labels]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure()
        plt.bar(x - width / 2, counts_gt, width, label="Ground Truth")
        plt.bar(x + width / 2, counts_pr, width, label="Predictions")
        plt.xlabel("Label ID")
        plt.ylabel("Count")
        plt.title(f"{exp} – Left: Ground Truth, Right: Generated Samples (SPR_BENCH)")
        plt.legend()
        plt.xticks(x, labels)
        fname = os.path.join(working_dir, f"SPR_BENCH_{exp}_LabelDist.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating label dist for {exp}: {e}")
        plt.close()

# -------------------------------------------------- quick metrics summary
for exp in exp_names:
    try:
        last_val = experiment_data[exp][dataset]["metrics"]["val"][-1]
        print(
            f"{exp}: final Macro-F1={last_val['macro_f1']:.3f}, "
            f"Accuracy(RGA)={last_val['RGA']:.3f}"
        )
    except Exception:
        pass
