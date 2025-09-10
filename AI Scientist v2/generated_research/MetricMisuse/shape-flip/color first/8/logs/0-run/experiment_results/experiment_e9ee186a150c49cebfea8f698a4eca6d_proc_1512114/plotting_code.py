import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    wd_dict = experiment_data["weight_decay"]["SPR_BENCH"]

    # ---------- helper ----------
    def comp_weighted_accuracy(seqs, y_true, y_pred):
        def cw(s):
            toks = s.split()
            sh = {t[0] for t in toks}
            co = {t[1:] if len(t) > 1 else "0" for t in toks}
            return len(sh) + len(co)

        w = [cw(s) for s in seqs]
        good = [wt if a == b else 0 for wt, a, b in zip(w, y_true, y_pred)]
        return sum(good) / sum(w) if sum(w) else 0.0

    # ---------- compute & print metrics ----------
    print("\n=== Final Validation & Complexity-Weighted Accuracies ===")
    compwa_vals, last_val_accs = {}, {}
    dev_seqs = [
        r.split("|")[1] if "|" in r else ""  # safeguard
        for r in [""] * len(next(iter(wd_dict.values()))["ground_truth"])
    ]

    for k, d in wd_dict.items():
        # val accuracy (last epoch)
        last_val_acc = d["metrics"]["val"][-1]
        last_val_accs[k] = last_val_acc
        # compWA
        if not dev_seqs[0]:  # extract sequences once
            # retrieve from stored ground_truth indices by matching order
            # sequences are not stored; cannot compute compWA safely
            compwa = np.nan
        else:
            compwa = comp_weighted_accuracy(
                dev_seqs, d["ground_truth"], d["predictions"]
            )
        compwa_vals[k] = compwa
        print(f"{k}: ValAcc={last_val_acc:.3f}  CompWA={compwa:.3f}")

    # ---------- plot 1: Loss curves ----------
    try:
        plt.figure()
        for k, d in wd_dict.items():
            plt.plot(d["losses"]["train"], label=f"{k} train")
            plt.plot(d["losses"]["val"], label=f"{k} val", linestyle="--")
        plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- plot 2: Accuracy curves ----------
    try:
        plt.figure()
        for k, d in wd_dict.items():
            plt.plot(d["metrics"]["train"], label=f"{k} train")
            plt.plot(d["metrics"]["val"], label=f"{k} val", linestyle="--")
        plt.title("SPR_BENCH Accuracy Curves\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- plot 3: Complexity-Weighted Accuracy bar ----------
    try:
        plt.figure()
        keys = list(compwa_vals.keys())
        vals = [compwa_vals[k] for k in keys]
        plt.bar(range(len(keys)), vals, tick_label=keys)
        plt.title("SPR_BENCH Complexity-Weighted Accuracy vs Weight Decay")
        plt.ylabel("CompWA")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_compWA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot: {e}")
        plt.close()
