import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    hid_dict = experiment_data["hidden_dim"]["SPR_BENCH"]
    hidden_dims = sorted([int(k) for k in hid_dict.keys()])  # [32,64,128,256]

    # --------- Per-hidden_dim curves -------------
    for hd in hidden_dims:
        try:
            rec = hid_dict[str(hd)]
            tr_loss = rec["losses"]["train"]
            vl_loss = rec["losses"]["val"]
            tr_acc = [m["acc"] for m in rec["metrics"]["train"]]
            vl_acc = [m["acc"] for m in rec["metrics"]["val"]]
            epochs = np.arange(1, len(tr_loss) + 1)

            plt.figure(figsize=(8, 4))
            # Left subplot: Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, vl_loss, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss")
            plt.legend()
            # Right subplot: Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(epochs, tr_acc, label="Train")
            plt.plot(epochs, vl_acc, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy")
            plt.legend()

            plt.suptitle(f"SPR_BENCH hidden_dim={hd}\nLeft: Loss, Right: Accuracy")
            fname = os.path.join(working_dir, f"SPR_BENCH_hidden{hd}_loss_acc.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating plot for hidden_dim={hd}: {e}")
            plt.close()

    # -------- Summary bar plot ----------------
    try:
        test_acc = [hid_dict[str(hd)]["test"]["acc"] for hd in hidden_dims]
        test_cowa = [hid_dict[str(hd)]["test"]["cowa"] for hd in hidden_dims]
        x = np.arange(len(hidden_dims))
        width = 0.35

        plt.figure(figsize=(6, 4))
        plt.bar(x - width / 2, test_acc, width, label="Accuracy")
        plt.bar(x + width / 2, test_cowa, width, label="COWA")
        plt.xticks(x, hidden_dims)
        plt.xlabel("hidden_dim")
        plt.ylabel("Metric")
        plt.title("SPR_BENCH Test Metrics")
        plt.legend()
        plt.suptitle("Test Performance per hidden_dim")
        fname = os.path.join(working_dir, "SPR_BENCH_test_summary.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    # --------- Print table --------------------
    print("hidden_dim | Test Loss | Test Acc | Test COWA")
    for hd in hidden_dims:
        rec = hid_dict[str(hd)]["test"]
        print(
            f"{hd:9d} | {rec['loss']:.4f}   | {rec['acc']:.3f}    | {rec['cowa']:.3f}"
        )
