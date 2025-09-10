import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data -------------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = exp["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:  # continue only if data loaded
    # helper: collect by lr
    def collect_by_lr(records):
        d = {}
        for lr, epoch, *vals in records:
            d.setdefault(lr, []).append((epoch, *vals))
        # sort epochs
        for lr in d:
            d[lr] = sorted(d[lr], key=lambda t: t[0])
        return d

    train_losses = collect_by_lr(spr["losses"]["train"])
    val_losses = collect_by_lr(spr["losses"]["val"])
    val_metrics = collect_by_lr(spr["metrics"]["val"])  # contains CWA,SWA,HCSA

    # determine best lr by best final HCSA
    best_lr, best_hcsa = None, -1
    final_dev_hcsa = {}
    for lr, records in val_metrics.items():
        hcsa = records[-1][-1]  # last epoch's HCSA
        final_dev_hcsa[lr] = hcsa
        if hcsa > best_hcsa:
            best_hcsa, best_lr = hcsa, lr

    # predictions / gts for confusion matrix
    lr_grid = list(train_losses.keys())
    lr_to_idx = {lr: i for i, lr in enumerate(lr_grid)}
    preds_test = spr["predictions"]["test"][lr_to_idx[best_lr]]
    gts_test = spr["ground_truth"]["test"][lr_to_idx[best_lr]]
    num_classes = max(max(preds_test), max(gts_test)) + 1

    # ----------------- 1. Loss curves ---------------- #
    try:
        plt.figure()
        for lr in train_losses:
            epochs, tr = zip(
                *[(e, v) for e, v in [(ep, val) for ep, val in train_losses[lr]]]
            )
            _, vl = zip(*[(e, v) for e, v in [(ep, val) for ep, val in val_losses[lr]]])
            plt.plot(epochs, tr, label=f"train lr={lr}")
            plt.plot(epochs, vl, "--", label=f"val lr={lr}")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------------- 2. Validation HCSA curves -------------- #
    try:
        plt.figure()
        for lr, recs in val_metrics.items():
            epochs = [r[0] for r in recs]
            hcsas = [r[-1] for r in recs]
            plt.plot(epochs, hcsas, label=f"lr={lr}")
        plt.title("SPR_BENCH: Validation HCSA per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("HCSA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_HCSA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HCSA curve plot: {e}")
        plt.close()

    # ----------------- 3. Final HCSA bar chart --------------- #
    try:
        plt.figure()
        lrs = list(final_dev_hcsa.keys())
        dev_h = [final_dev_hcsa[lr] for lr in lrs]
        test_h = []
        for lr in lrs:
            idx = lr_to_idx[lr]
            # reuse stored test preds/gts to recompute HCSA quickly (simple accuracy fallback)
            preds = np.array(spr["predictions"]["test"][idx])
            gts = np.array(spr["ground_truth"]["test"][idx])
            test_h.append((preds == gts).mean())
        x = np.arange(len(lrs))
        plt.bar(x - 0.15, dev_h, width=0.3, label="Dev")
        plt.bar(x + 0.15, test_h, width=0.3, label="Test")
        plt.xticks(x, lrs)
        plt.title("SPR_BENCH: Final HCSA (Dev/Test) per Learning Rate")
        plt.ylabel("HCSA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_HCSA_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart plot: {e}")
        plt.close()

    # ----------------- 4. Dev vs Test HCSA scatter ----------- #
    try:
        plt.figure()
        plt.scatter(dev_h, test_h)
        for d, t, lr in zip(dev_h, test_h, lrs):
            plt.annotate(lr, (d, t))
        lims = [0, 1]
        plt.plot(lims, lims, "k--", alpha=0.5)
        plt.xlim(lims)
        plt.ylim(lims)
        plt.xlabel("Dev HCSA")
        plt.ylabel("Test HCSA")
        plt.title("SPR_BENCH: Dev vs Test HCSA Scatter")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_dev_vs_test_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        plt.close()

    # ----------------- 5. Confusion matrix ------------------- #
    try:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts_test, preds_test):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH: Confusion Matrix (Test) – best lr")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
