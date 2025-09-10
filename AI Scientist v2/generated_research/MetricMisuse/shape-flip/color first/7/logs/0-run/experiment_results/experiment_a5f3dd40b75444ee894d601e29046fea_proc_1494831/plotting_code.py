import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    for dname, ds in experiment_data.items():
        epochs = np.arange(1, len(ds["losses"]["train"]) + 1)

        # 1) Loss curves -------------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, ds["losses"]["train"], label="Train")
            plt.plot(epochs, ds["losses"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname} Loss Curves")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dname}: {e}")
            plt.close()

        # 2) Accuracy curves ---------------------------------------------------
        try:
            plt.figure()
            tr_acc = [m["acc"] for m in ds["metrics"]["train"]]
            val_acc = [m["acc"] for m in ds["metrics"]["val"]]
            plt.plot(epochs, tr_acc, label="Train")
            plt.plot(epochs, val_acc, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} Accuracy Curves")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_accuracy_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {dname}: {e}")
            plt.close()

        # 3) Complexity-Weighted Accuracy -------------------------------------
        try:
            plt.figure()
            val_cwa = [m["CompWA"] for m in ds["metrics"]["val"]]
            plt.plot(epochs, val_cwa, label="Validation CWA")
            plt.xlabel("Epoch")
            plt.ylabel("Comp-Weighted Acc")
            plt.title(f"{dname} Complexity-Weighted Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_cwa_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating CWA plot for {dname}: {e}")
            plt.close()

        # ---------- evaluation metrics ---------------------------------------
        try:
            preds = np.array(ds["predictions"])
            gts = np.array(ds["ground_truth"])
            seqs = np.array(ds["sequences"])

            test_acc = (preds == gts).mean()

            # helper functions (same as training script)
            def count_color_variety(sequence: str) -> int:
                return len({tok[1:] for tok in sequence.split() if len(tok) > 1})

            def count_shape_variety(sequence: str) -> int:
                return len({tok[0] for tok in sequence.split() if tok})

            def complexity_weight(seq: str) -> int:
                return count_color_variety(seq) + count_shape_variety(seq)

            weights = np.array([complexity_weight(s) for s in seqs])
            cwa = (weights * (preds == gts)).sum() / max(1, weights.sum())

            print(f"[{dname}] Test Accuracy: {test_acc:.3f} | Test CWA: {cwa:.3f}")
        except Exception as e:
            print(f"Error computing evaluation metrics for {dname}: {e}")
