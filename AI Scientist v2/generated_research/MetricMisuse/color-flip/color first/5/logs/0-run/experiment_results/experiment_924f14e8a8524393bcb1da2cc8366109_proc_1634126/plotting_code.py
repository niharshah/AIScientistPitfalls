import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
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
    bs_dict = experiment_data["batch_size"]["SPR_BENCH"]
    # store final-epoch test metrics to summarise later
    summary = {"bs": [], "ACC": [], "CWA": [], "SWA": [], "CompWA": []}

    for bs, rec in bs_dict.items():
        # ---------- loss curve ----------
        try:
            plt.figure()
            epochs = np.arange(1, len(rec["losses"]["train"]) + 1)
            plt.plot(epochs, rec["losses"]["train"], label="Train")
            plt.plot(epochs, rec["losses"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"SPR_BENCH – Loss vs Epoch (batch_size={bs})")
            plt.legend()
            fname = f"SPR_BENCH_loss_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for bs={bs}: {e}")
            plt.close()

        # ---------- accuracy curve ----------
        try:
            plt.figure()
            val_metrics = rec["metrics"]["val"]
            epochs = [m["epoch"] for m in val_metrics]
            accs = [m["acc"] for m in val_metrics]
            plt.plot(epochs, accs, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title(f"SPR_BENCH – Accuracy vs Epoch (batch_size={bs})")
            fname = f"SPR_BENCH_valACC_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for bs={bs}: {e}")
            plt.close()

        # ---------- gather summary ----------
        t_m = rec["metrics"]["test"]
        summary["bs"].append(int(bs))
        summary["ACC"].append(t_m["acc"])
        summary["CWA"].append(t_m["cwa"])
        summary["SWA"].append(t_m["swa"])
        summary["CompWA"].append(t_m["compwa"])

    # ---------- summary bar plot ----------
    try:
        xs = np.arange(len(summary["bs"]))
        width = 0.2
        plt.figure(figsize=(8, 4))
        for i, key in enumerate(["ACC", "CWA", "SWA", "CompWA"]):
            plt.bar(xs + i * width, summary[key], width, label=key)
        plt.xticks(xs + 1.5 * width, summary["bs"])
        plt.xlabel("Batch Size")
        plt.ylabel("Metric Value")
        plt.title("SPR_BENCH – Test Metrics vs Batch Size")
        plt.legend()
        fname = "SPR_BENCH_test_metrics_by_bs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    # ---------- print numeric summary ----------
    print("\nTest-set metrics by batch size")
    for i in range(len(summary["bs"])):
        print(
            f"bs={summary['bs'][i]:3d} | "
            f"ACC={summary['ACC'][i]:.3f} | "
            f"CWA={summary['CWA'][i]:.3f} | "
            f"SWA={summary['SWA'][i]:.3f} | "
            f"CompWA={summary['CompWA'][i]:.3f}"
        )
