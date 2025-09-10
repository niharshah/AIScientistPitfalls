import matplotlib.pyplot as plt
import numpy as np
import os

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

if experiment_data:
    ls_dict = experiment_data.get("label_smoothing", {})
    # store test metrics for summary
    test_metrics = {"ls": [], "CRWA": [], "SWA": [], "CWA": []}

    for ls_str, rec in ls_dict.items():
        try:
            ls = float(ls_str)
            train_losses = rec["losses"]["train"]
            val_losses = rec["losses"]["val"]
            # extract CRWA across epochs
            val_crwa = [m["CRWA"] for m in rec["metrics"]["val"]]
            epochs = np.arange(1, len(train_losses) + 1)

            plt.figure(figsize=(10, 4))
            # Left subplot: losses
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_losses, label="Train")
            plt.plot(epochs, val_losses, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.title("Loss")

            # Right subplot: CRWA
            plt.subplot(1, 2, 2)
            plt.plot(epochs, val_crwa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("CRWA")
            plt.title("CRWA")

            plt.suptitle(
                f"SPR_BENCH | label_smoothing={ls}  \nLeft: Loss (Train/Val), Right: CRWA (Val)"
            )
            fname = os.path.join(working_dir, f"SPR_BENCH_ls{ls}_loss_metric.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error creating plot for ls={ls_str}: {e}")
            plt.close()

        # collect test metrics
        try:
            tm = rec["metrics"]["test"]
            test_metrics["ls"].append(float(ls_str))
            test_metrics["CRWA"].append(tm["CRWA"])
            test_metrics["SWA"].append(tm["SWA"])
            test_metrics["CWA"].append(tm["CWA"])
        except Exception as e:
            print(f"Error extracting test metrics for ls={ls_str}: {e}")

    # ---------- summary figure ----------
    try:
        x = np.arange(len(test_metrics["ls"]))
        width = 0.25
        plt.figure(figsize=(8, 4))
        plt.bar(x - width, test_metrics["CRWA"], width, label="CRWA")
        plt.bar(x, test_metrics["SWA"], width, label="SWA")
        plt.bar(x + width, test_metrics["CWA"], width, label="CWA")
        plt.xticks(x, [str(ls) for ls in test_metrics["ls"]])
        plt.xlabel("Label Smoothing")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Test Metrics vs. Label Smoothing")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_summary_metrics.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating summary figure: {e}")
        plt.close()
