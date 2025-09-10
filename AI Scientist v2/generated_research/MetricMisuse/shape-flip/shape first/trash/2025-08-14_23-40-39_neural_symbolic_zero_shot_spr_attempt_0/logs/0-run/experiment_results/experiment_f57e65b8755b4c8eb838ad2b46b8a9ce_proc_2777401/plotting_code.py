import matplotlib.pyplot as plt
import numpy as np
import os

# working directory setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    metrics = experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]
    bps_dict = metrics["val_bps"]
    train_loss_dict = metrics["train_loss"]
    val_loss_dict = metrics["val_loss"]
    batch_sizes = sorted(train_loss_dict.keys())

    # 1. Train / Val loss curves
    try:
        plt.figure()
        for bs in batch_sizes:
            epochs = np.arange(1, len(train_loss_dict[bs]) + 1)
            plt.plot(
                epochs, train_loss_dict[bs], label=f"Train bs={bs}", linestyle="--"
            )
            plt.plot(epochs, val_loss_dict[bs], label=f"Val bs={bs}")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. Validation BPS curves
    try:
        plt.figure()
        for bs in batch_sizes:
            epochs = np.arange(1, len(bps_dict[bs]) + 1)
            plt.plot(epochs, bps_dict[bs], label=f"bs={bs}")
        plt.title("SPR_BENCH: Validation BPS Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("BPS")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_val_bps_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating BPS curve plot: {e}")
        plt.close()

    # 3. Final DEV vs TEST BPS scatter
    try:
        dev_final = experiment_data["batch_size_sweep"]["SPR_BENCH"]["predictions"][
            "dev"
        ]
        test_final = experiment_data["batch_size_sweep"]["SPR_BENCH"]["predictions"][
            "test"
        ]
        # convert predictions to stored bps; they are not directly stored, so grab metrics at last epoch
        dev_bps = []
        test_bps = []
        for bs in batch_sizes:
            dev_bps.append(bps_dict[bs][-1])
            # test BPS stored in metrics 'val_bps' only for val; compute from saved test metrics dict
            # fallback: read last evaluated test_bps in metrics if stored
            test_metric = (
                experiment_data["batch_size_sweep"]["SPR_BENCH"]["metrics"]
                .get("test_bps", {})
                .get(bs, None)
            )
            if test_metric:
                test_bps.append(test_metric[-1])
            else:
                test_bps.append(np.nan)

        plt.figure()
        plt.scatter(dev_bps, test_bps)
        for i, bs in enumerate(batch_sizes):
            plt.annotate(str(bs), (dev_bps[i], test_bps[i]))
        plt.title("SPR_BENCH: Final DEV vs TEST BPS")
        plt.xlabel("DEV BPS")
        plt.ylabel("TEST BPS")
        save_path = os.path.join(working_dir, "SPR_BENCH_dev_vs_test_bps.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating DEV vs TEST BPS plot: {e}")
        plt.close()

    # Print summary table
    print("\nBatch Size | Final DEV BPS | Final TEST BPS")
    for i, bs in enumerate(batch_sizes):
        test_val = test_bps[i] if not np.isnan(test_bps[i]) else "NA"
        print(f"{bs:10} | {dev_bps[i]:13.4f} | {test_val}")
