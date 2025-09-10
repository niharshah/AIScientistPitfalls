import matplotlib.pyplot as plt
import numpy as np
import os

# ----- paths -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

lr_dict = experiment_data.get("learning_rate", {})
if not lr_dict:
    print("No learning-rate results found in experiment_data.npy")


# helper: unpack list[(epoch,val)] -> two lists
def unzip(pairs):
    if not pairs:
        return [], []
    ep, val = zip(*pairs)
    return list(ep), list(val)


# ------------- per-LR plots -------------
for lr_key, lr_res in lr_dict.items():
    losses = lr_res["SPR_BENCH"]["losses"]
    metrics = lr_res["SPR_BENCH"]["metrics"]["val"]
    # loss curve
    try:
        ep_tr, tr = unzip(losses["train"])
        ep_va, va = unzip(losses["val"])
        plt.figure()
        plt.plot(ep_tr, tr, label="Train")
        plt.plot(ep_va, va, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Loss Curve (lr={lr_key})")
        plt.legend()
        fname = f"SPR_BENCH_loss_curve_lr_{lr_key.replace('.', 'p')}.png"
        plt.savefig(os.path.join(working_dir, fname))
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for lr={lr_key}: {e}")
        plt.close()

    # metric curve (CWA/SWA/EWA) – validation only
    try:
        epochs = []
        cwa, swa, ewa = [], [], []
        for ep, d in metrics:
            epochs.append(ep)
            cwa.append(d["CWA"])
            swa.append(d["SWA"])
            ewa.append(d["EWA"])
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, ewa, label="EWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(f"SPR_BENCH Metric Curve (lr={lr_key})")
        plt.legend()
        fname = f"SPR_BENCH_metric_curve_lr_{lr_key.replace('.', 'p')}.png"
        plt.savefig(os.path.join(working_dir, fname))
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for lr={lr_key}: {e}")
        plt.close()

# ------------- comparison bar chart -------------
try:
    lrs, cwa_best, swa_best, ewa_best = [], [], [], []
    for lr_key, lr_res in lr_dict.items():
        preds = lr_res["SPR_BENCH"]["predictions"]
        gt = lr_res["SPR_BENCH"]["ground_truth"]

        # reuse metrics functions stored in experiment_data? compute quickly here by ratio
        # We stored test CWA/SWA/EWA only via prints, so recompute:
        def cwa(seq, y_t, y_p):
            from collections import Counter

            def count_color_variety(seq):
                return len(set(t[1] for t in seq.split() if len(t) > 1))

            weights = [count_color_variety(s) for s in seq]
            correct = [w if t == p else 0 for w, t, p in zip(weights, y_t, y_p)]
            return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

        def swa(seq, y_t, y_p):
            def count_shape_variety(seq):
                return len(set(t[0] for t in seq.split() if t))

            weights = [count_shape_variety(s) for s in seq]
            correct = [w if t == p else 0 for w, t, p in zip(weights, y_t, y_p)]
            return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

        def ewa(seq, y_t, y_p):
            import math
            from collections import Counter

            def entropy_weight(seq):
                toks = seq.split()
                total = len(toks)
                if not toks:
                    return 0.0
                freqs = Counter(toks)
                return -sum((c / total) * math.log2(c / total) for c in freqs.values())

            weights = [entropy_weight(s) for s in seq]
            correct = [w if t == p else 0 for w, t, p in zip(weights, y_t, y_p)]
            return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

        seqs = lr_res["SPR_BENCH"]["ground_truth"]  # same length, just reuse
        lrs.append(lr_key)
        cwa_best.append(cwa(seqs, gt, preds))
        swa_best.append(swa(seqs, gt, preds))
        ewa_best.append(ewa(seqs, gt, preds))

    x = np.arange(len(lrs))
    width = 0.25
    plt.figure(figsize=(6, 4))
    plt.bar(x - width, cwa_best, width, label="CWA")
    plt.bar(x, swa_best, width, label="SWA")
    plt.bar(x + width, ewa_best, width, label="EWA")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Weighted Accuracy")
    plt.title("SPR_BENCH Test Metrics by Learning Rate")
    plt.xticks(x, lrs, rotation=45)
    plt.legend()
    fname = "SPR_BENCH_test_metric_comparison.png"
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating comparison bar chart: {e}")
    plt.close()

# ------------- optional console summary -------------
print("\nFinal Test Metrics")
for lr, c, s, e in zip(lrs, cwa_best, swa_best, ewa_best):
    print(f"lr={lr:>6}: CWA={c:.4f}  SWA={s:.4f}  EWA={e:.4f}")
