import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs_dict = experiment_data["EPOCHS"]["SPR_BENCH"]
    run_keys = sorted(
        [k for k in runs_dict if k.startswith("run_")],
        key=lambda s: int(s.split("_")[-1]),
    )

    # --------------------------------------------------------------
    # 1. training/validation loss curves
    # --------------------------------------------------------------
    for rk in run_keys:
        try:
            losses = runs_dict[rk]["losses"]
            epochs = np.arange(1, len(losses["train"]) + 1)
            plt.figure()
            plt.plot(epochs, losses["train"], label="Train Loss")
            plt.plot(epochs, losses["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{rk}: Loss Curves (SPR_BENCH)")
            plt.legend()
            fname = f"spr_bench_{rk}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {rk}: {e}")
            plt.close()

    # --------------------------------------------------------------
    # 2. validation HWA curves
    # --------------------------------------------------------------
    for rk in run_keys:
        try:
            hwa = [m["hwa"] for m in runs_dict[rk]["metrics"]["val"]]
            epochs = np.arange(1, len(hwa) + 1)
            plt.figure()
            plt.plot(epochs, hwa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Harmonic Weighted Acc")
            plt.title(f"{rk}: Validation HWA (SPR_BENCH)")
            fname = f"spr_bench_{rk}_hwa_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating HWA plot for {rk}: {e}")
            plt.close()

    # --------------------------------------------------------------
    # 3. final HWA per run (bar chart)
    # --------------------------------------------------------------
    try:
        final_hwa = [runs_dict[rk]["final_val_hwa"] for rk in run_keys]
        plt.figure()
        plt.bar(run_keys, final_hwa)
        plt.ylabel("Final Val HWA")
        plt.title("SPR_BENCH: Final Validation HWA by Epoch Setting")
        plt.xticks(rotation=45)
        fname = "spr_bench_final_hwa_bar.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating final HWA bar chart: {e}")
        plt.close()

    # --------------------------------------------------------------
    # 4. confusion matrix on test set for best run
    # --------------------------------------------------------------
    try:
        best_r = runs_dict["best_run"] if "best_run" in runs_dict else None
        if best_r is not None:
            preds = runs_dict["predictions"]
            golds = runs_dict["ground_truth"]
            labels = sorted(list(set(golds) | set(preds)))
            lbl2idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for g, p in zip(golds, preds):
                cm[lbl2idx[g], lbl2idx[p]] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
            plt.title(
                "SPR_BENCH Test Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            plt.tight_layout()
            fname = "spr_bench_confusion_matrix_best.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        else:
            print("No best_run info for confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------------------------------------------------------------
    # print stored test metrics
    # --------------------------------------------------------------
    try:
        test_metrics = runs_dict["metrics_test"]
        print(
            f"Stored TEST metrics -> CWA: {test_metrics['cwa']:.3f}, "
            f"SWA: {test_metrics['swa']:.3f}, "
            f"HWA: {test_metrics['hwa']:.3f}"
        )
    except Exception as e:
        print(f"Error printing test metrics: {e}")
