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
    # ---------- unpack data ----------
    losses_tr = experiment_data["batch_size_tuning"]["losses"]["train"]
    losses_val = experiment_data["batch_size_tuning"]["losses"]["val"]
    metrics_val = experiment_data["batch_size_tuning"]["metrics"]["val"]

    batch_sizes = [16, 32, 64, 128]  # known from training script
    epochs = 5
    n_bs = len(batch_sizes)

    # reshape helpers
    lt = np.array(losses_tr).reshape(n_bs, epochs)
    lv = np.array(losses_val).reshape(n_bs, epochs)

    # accuracy / hwa matrices
    acc = np.array(
        [
            [m["acc"] for m in metrics_val[i * epochs : (i + 1) * epochs]]
            for i in range(n_bs)
        ]
    )
    hwa = np.array(
        [
            [m["hwa"] for m in metrics_val[i * epochs : (i + 1) * epochs]]
            for i in range(n_bs)
        ]
    )

    # ---------- plotting ----------
    # 1. Loss curves
    try:
        plt.figure()
        for i, bs in enumerate(batch_sizes):
            plt.plot(range(1, epochs + 1), lt[i], "--", label=f"train bs={bs}")
            plt.plot(range(1, epochs + 1), lv[i], "-", label=f"val bs={bs}")
        plt.title("Training & Validation Losses\nDataset: Synthetic Shapes-Colors")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "synthetic_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. Validation accuracy curves
    try:
        plt.figure()
        for i, bs in enumerate(batch_sizes):
            plt.plot(range(1, epochs + 1), acc[i], label=f"bs={bs}")
        plt.title("Validation Accuracy vs. Epoch\nDataset: Synthetic Shapes-Colors")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "synthetic_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3. Validation HWA curves
    try:
        plt.figure()
        for i, bs in enumerate(batch_sizes):
            plt.plot(range(1, epochs + 1), hwa[i], label=f"bs={bs}")
        plt.title(
            "Validation Harmonic Weighted Accuracy vs. Epoch\nDataset: Synthetic Shapes-Colors"
        )
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        fname = os.path.join(working_dir, "synthetic_hwa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # 4. Best HWA per batch size bar chart
    try:
        best_hwa = hwa.max(axis=1)
        plt.figure()
        plt.bar([str(bs) for bs in batch_sizes], best_hwa)
        plt.title(
            "Best Harmonic Weighted Accuracy by Batch Size\nDataset: Synthetic Shapes-Colors"
        )
        plt.xlabel("Batch Size")
        plt.ylabel("Best HWA")
        fname = os.path.join(working_dir, "synthetic_best_hwa_by_bs.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating best HWA bar chart: {e}")
        plt.close()
