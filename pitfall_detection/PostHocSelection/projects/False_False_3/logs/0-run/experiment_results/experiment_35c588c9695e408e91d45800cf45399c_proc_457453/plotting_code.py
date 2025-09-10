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
    exp = experiment_data["label_smoothing"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    sm_vals = exp["smoothing_values"]
    val_loss = exp["losses"]["val"]
    val_mets = exp["metrics"]["val"]
    swa_vals = [d["SWA"] for d in val_mets]
    cwa_vals = [d["CWA"] for d in val_mets]
    hwa_vals = [d["HWA"] for d in val_mets]
    preds = exp["predictions"]
    gts = exp["ground_truth"]
    best_sm = exp["best_smoothing"]
    tst_mets = exp["test_metrics"]
    tst_loss = exp["test_loss"]

    # ---------- 1) Validation loss vs smoothing ----------
    try:
        plt.figure()
        plt.plot(sm_vals, val_loss, marker="o")
        plt.title(
            "SPR_BENCH: Validation Loss vs Label Smoothing\nLeft: coef, Right: loss"
        )
        plt.xlabel("Label Smoothing Coefficient")
        plt.ylabel("Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_vs_smoothing.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating val-loss plot: {e}")
        plt.close()

    # ---------- 2) Validation weighted accuracies ----------
    try:
        plt.figure()
        plt.plot(sm_vals, swa_vals, marker="o", label="SWA")
        plt.plot(sm_vals, cwa_vals, marker="s", label="CWA")
        plt.plot(sm_vals, hwa_vals, marker="^", label="HWA")
        plt.title(
            "SPR_BENCH: Weighted Accuracies vs Label Smoothing\nLeft: coef, Right: accuracy"
        )
        plt.xlabel("Label Smoothing Coefficient")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(
                working_dir, "SPR_BENCH_val_weighted_accuracies_vs_smoothing.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating val-acc plot: {e}")
        plt.close()

    # ---------- 3) Confusion matrix on test ----------
    try:
        labels = sorted(set(gts) | set(preds))
        lbl2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[lbl2idx[t], lbl2idx[p]] += 1

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.title("SPR_BENCH: Test Confusion Matrix\nRows: True, Cols: Pred")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion-matrix plot: {e}")
        plt.close()

    # ---------- 4) Test weighted accuracies bar ----------
    try:
        plt.figure()
        metrics = ["SWA", "CWA", "HWA"]
        values = [tst_mets[m] for m in metrics]
        plt.bar(metrics, values, color=["tab:orange", "tab:green", "tab:purple"])
        plt.ylim(0, 1)
        plt.title(f"SPR_BENCH: Test Weighted Accuracies (best smoothing={best_sm:.2f})")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_test_weighted_accuracies_bar.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating test-acc bar plot: {e}")
        plt.close()

    # ---------- print summary ----------
    print(f"Best smoothing on dev: {best_sm:.2f} | dev HWA={max(hwa_vals):.4f}")
    print(
        f'TEST | loss={tst_loss:.4f} | SWA={tst_mets["SWA"]:.4f} | CWA={tst_mets["CWA"]:.4f} | HWA={tst_mets["HWA"]:.4f}'
    )
