import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    wd_runs = experiment_data.get("weight_decay", {})
    # Collect summary metrics
    summary = {}
    for tag, run in wd_runs.items():
        val_accs = [m["acc"] for m in run["metrics"]["val"]]
        best_val = max(val_accs)
        preds, gts = run["predictions"], run["ground_truth"]
        test_acc = sum(int(p == g) for p, g in zip(preds, gts)) / max(1, len(gts))
        summary[tag] = {"best_val_acc": best_val, "test_acc": test_acc}

    # 1) Loss curves
    try:
        plt.figure()
        for tag, run in wd_runs.items():
            plt.plot(run["losses"]["train"], label=f"{tag}-train")
            plt.plot(run["losses"]["val"], linestyle="--", label=f"{tag}-val")
        plt.title("SPR Dataset – Training vs Validation Loss (Weight Decay Sweep)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curves_weight_decay.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        for tag, run in wd_runs.items():
            tr = [m["acc"] for m in run["metrics"]["train"]]
            va = [m["acc"] for m in run["metrics"]["val"]]
            plt.plot(tr, label=f"{tag}-train")
            plt.plot(va, linestyle="--", label=f"{tag}-val")
        plt.title("SPR Dataset – Training vs Validation Accuracy (Weight Decay Sweep)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_accuracy_curves_weight_decay.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3) Complexity-weighted accuracy curves
    try:
        plt.figure()
        for tag, run in wd_runs.items():
            tr = [m["cowa"] for m in run["metrics"]["train"]]
            va = [m["cowa"] for m in run["metrics"]["val"]]
            plt.plot(tr, label=f"{tag}-train")
            plt.plot(va, linestyle="--", label=f"{tag}-val")
        plt.title("SPR Dataset – Training vs Validation COWA (Weight Decay Sweep)")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_cowa_curves_weight_decay.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating COWA plot: {e}")
        plt.close()

    # Print summary
    print("\nBest Validation Accuracies and Test Accuracies per Weight Decay")
    best_tag, best_val = None, -1
    for tag, vals in summary.items():
        print(
            f"{tag}: best_val_acc={vals['best_val_acc']:.3f} | test_acc={vals['test_acc']:.3f}"
        )
        if vals["best_val_acc"] > best_val:
            best_val, best_tag = vals["best_val_acc"], tag
    print(
        f"\n=> Best weight_decay setting based on val accuracy: {best_tag} (val_acc={best_val:.3f})"
    )
