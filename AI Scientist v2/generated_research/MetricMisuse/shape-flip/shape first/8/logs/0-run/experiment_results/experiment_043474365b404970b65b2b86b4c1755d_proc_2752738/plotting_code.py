import matplotlib.pyplot as plt
import numpy as np
import os

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

if experiment_data is not None:
    spr_data = experiment_data["batch_size"]["SPR_BENCH"]
    batch_sizes = sorted(spr_data.keys())

    # gather arrays
    epochs = len(next(iter(spr_data.values()))["metrics"]["train_acc"])
    ep_idx = np.arange(1, epochs + 1)
    train_acc = {bs: spr_data[bs]["metrics"]["train_acc"] for bs in batch_sizes}
    val_acc = {bs: spr_data[bs]["metrics"]["val_acc"] for bs in batch_sizes}
    train_loss = {bs: spr_data[bs]["losses"]["train"] for bs in batch_sizes}
    test_acc = {bs: spr_data[bs]["test"]["acc"] for bs in batch_sizes}
    test_ura = {bs: spr_data[bs]["test"]["ura"] for bs in batch_sizes}

    # ---------- PLOT 1: accuracy curves ----------
    try:
        plt.figure()
        for bs in batch_sizes:
            plt.plot(ep_idx, train_acc[bs], label=f"train bs={bs}", linestyle="-")
            plt.plot(ep_idx, val_acc[bs], label=f"val   bs={bs}", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Dataset: SPR_BENCH\nTrain vs Validation Accuracy across Epochs")
        plt.legend(ncol=2, fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_acc_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- PLOT 2: loss curves ----------
    try:
        plt.figure()
        for bs in batch_sizes:
            plt.plot(ep_idx, train_loss[bs], label=f"bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Dataset: SPR_BENCH\nTraining Loss across Epochs")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- PLOT 3: test accuracy bar ----------
    try:
        plt.figure()
        plt.bar(
            range(len(batch_sizes)),
            [test_acc[bs] for bs in batch_sizes],
            tick_label=batch_sizes,
        )
        plt.ylim(0, 1)
        plt.xlabel("Batch Size")
        plt.ylabel("Test Accuracy")
        plt.title("Dataset: SPR_BENCH\nFinal Test Accuracy vs Batch Size")
        fname = os.path.join(working_dir, "SPR_BENCH_test_acc_vs_bs.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating test-acc bar: {e}")
        plt.close()

    # ---------- PLOT 4: URA bar ----------
    try:
        plt.figure()
        plt.bar(
            range(len(batch_sizes)),
            [test_ura[bs] for bs in batch_sizes],
            tick_label=batch_sizes,
            color="orange",
        )
        plt.ylim(0, 1)
        plt.xlabel("Batch Size")
        plt.ylabel("Unseen Rule Acc (URA)")
        plt.title("Dataset: SPR_BENCH\nTest URA vs Batch Size")
        fname = os.path.join(working_dir, "SPR_BENCH_test_ura_vs_bs.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating URA bar: {e}")
        plt.close()

    # ---------- print summary ----------
    print("\n=== Final Test Metrics ===")
    for bs in batch_sizes:
        print(f"bs={bs:3d} | acc={test_acc[bs]:.3f} | ura={test_ura[bs]:.3f}")
