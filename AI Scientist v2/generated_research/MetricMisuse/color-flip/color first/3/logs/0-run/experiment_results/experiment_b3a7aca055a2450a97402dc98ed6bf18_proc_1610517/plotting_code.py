import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 1. Load experiment data                                            #
# ------------------------------------------------------------------ #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}


def unpack(list_of_tuples, idx):
    return [t[idx] for t in list_of_tuples]


# ------------------------------------------------------------------ #
# 2. Iterate through datasets and plot                               #
# ------------------------------------------------------------------ #
for dname, dct in exp.items():
    # -------------------- a. loss curves --------------------------- #
    try:
        plt.figure()
        tr_epochs = unpack(dct["losses"]["train"], 0)
        tr_loss = unpack(dct["losses"]["train"], 1)
        v_epochs = unpack(dct["losses"]["val"], 0)
        v_loss = unpack(dct["losses"]["val"], 1)
        plt.plot(tr_epochs, tr_loss, "--", label="Train")
        plt.plot(v_epochs, v_loss, "-", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"{dname}: Train vs. Val Loss\n(Standard sequence classification)")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # -------------------- b. metric curves ------------------------- #
    try:
        metrics_val = dct["metrics"]["val"]
        if metrics_val:
            epochs = unpack(metrics_val, 0)
            cwa = unpack(metrics_val, 1)
            swa = unpack(metrics_val, 2)
            hcs = unpack(metrics_val, 3)
            snwa = unpack(metrics_val, 4)

            fig, axs = plt.subplots(2, 2, figsize=(8, 6))
            axs = axs.flatten()
            for ax, data, ttl in zip(
                axs, [cwa, swa, hcs, snwa], ["CWA", "SWA", "HCSA", "SNWA"]
            ):
                ax.plot(epochs, data, "-o", ms=3)
                ax.set_xlabel("Epoch")
                ax.set_title(ttl)
            fig.suptitle(
                f"{dname}: Validation Metrics\n(Left-Top→Right-Bottom: CWA, SWA, HCSA, SNWA)"
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = os.path.join(working_dir, f"{dname}_val_metric_curves.png")
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating metric plot for {dname}: {e}")
        plt.close()

    # -------------------- c. dev vs. test accuracy ----------------- #
    try:
        for split in ["dev", "test"]:
            preds = np.array(dct["predictions"].get(split, []))
            gts = np.array(dct["ground_truth"].get(split, []))
            acc = (preds == gts).mean() if preds.size else np.nan
            dct.setdefault("acc", {})[split] = acc
        acc_dev, acc_test = dct["acc"]["dev"], dct["acc"]["test"]

        plt.figure()
        plt.bar(["Dev", "Test"], [acc_dev, acc_test], color=["steelblue", "orange"])
        plt.ylabel("Accuracy")
        plt.title(f"{dname}: Dev vs. Test Accuracy\n(Simple class agreement)")
        fname = os.path.join(working_dir, f"{dname}_dev_vs_test_accuracy.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating accuracy bar chart for {dname}: {e}")
        plt.close()

# ------------------------------------------------------------------ #
# 3. Print summary accuracies                                        #
# ------------------------------------------------------------------ #
for dname, dct in exp.items():
    dev_acc = dct.get("acc", {}).get("dev", float("nan"))
    test_acc = dct.get("acc", {}).get("test", float("nan"))
    print(f"{dname}: Dev Accuracy={dev_acc:.3f}, Test Accuracy={test_acc:.3f}")
