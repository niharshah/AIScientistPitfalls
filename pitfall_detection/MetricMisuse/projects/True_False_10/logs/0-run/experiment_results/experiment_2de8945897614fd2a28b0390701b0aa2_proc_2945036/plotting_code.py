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
    ed = experiment_data["batch_size_tuning"]["SPR_BENCH"]
    bs_list = ed["batch_sizes"]
    train_losses = ed["losses"]["train"]  # list of lists [bs][epoch]
    val_losses = ed["losses"]["val"]  # list of lists [bs][epoch]
    val_metrics = ed["metrics"]["val"]  # list [bs] of list[epoch]{}
    test_metrics = ed["metrics"]["test"]  # list [bs] of {}
    epochs = np.arange(1, len(train_losses[0]) + 1)

    # 1) Loss curves ----------------------------------------------------------
    try:
        plt.figure()
        for bs, tr, vl in zip(bs_list, train_losses, val_losses):
            plt.plot(epochs, tr, "--", label=f"train bs={bs}")
            plt.plot(epochs, vl, "-", label=f"val   bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_loss_curves_batch_size_tuning.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2) Validation CRWA curves ----------------------------------------------
    try:
        plt.figure()
        for bs, m in zip(bs_list, val_metrics):
            crwa = [ep["CRWA"] for ep in m]
            plt.plot(epochs, crwa, label=f"bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("CRWA")
        plt.title("SPR_BENCH: Validation CRWA over Epochs")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_val_CRWA_curves_batch_size_tuning.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating CRWA curve plot: {e}")
        plt.close()

    # 3) Test metric comparison ----------------------------------------------
    try:
        idx = np.arange(len(bs_list))
        width = 0.25
        crwa = [m["CRWA"] for m in test_metrics]
        swa = [m["SWA"] for m in test_metrics]
        cwa = [m["CWA"] for m in test_metrics]

        plt.figure(figsize=(8, 4))
        plt.bar(idx - width, crwa, width, label="CRWA")
        plt.bar(idx, swa, width, label="SWA")
        plt.bar(idx + width, cwa, width, label="CWA")
        plt.xticks(idx, bs_list)
        plt.xlabel("Batch Size")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Test Metrics vs Batch Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_test_metrics_batch_size_tuning.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot: {e}")
        plt.close()

    # -------- Console summary ------------------------------------------------
    print("\n=== Test Metrics ===")
    for bs, m in zip(bs_list, test_metrics):
        print(
            f"bs={bs:3}: CRWA={m['CRWA']:.4f} | SWA={m['SWA']:.4f} | CWA={m['CWA']:.4f}"
        )
