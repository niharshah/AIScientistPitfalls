import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["batch_size_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    batch_sizes = ed["batch_sizes"]
    train_acc = ed["metrics"]["train_acc"]  # list(len=batch_sizes)[epochs]
    val_acc = ed["metrics"]["val_acc"]
    val_ura = ed["metrics"]["val_ura"]
    train_loss = ed["losses"]["train"]
    test_acc = ed["metrics"]["test_acc"]  # list of scalars
    test_ura = ed["metrics"]["test_ura"]  # list of scalars
    epochs = np.arange(1, len(train_acc[0]) + 1)

    # 1. Train vs Val Accuracy
    try:
        plt.figure()
        for bs, ta, va in zip(batch_sizes, train_acc, val_acc):
            plt.plot(epochs, ta, "--", label=f"Train bs={bs}")
            plt.plot(epochs, va, "-", label=f"Val bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH: Training vs Validation Accuracy\nLeft: Train (dashed), Right: Validation (solid)"
        )
        plt.legend(fontsize=6)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2. Validation URA
    try:
        plt.figure()
        for bs, ura in zip(batch_sizes, val_ura):
            plt.plot(epochs, ura, label=f"bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("URA")
        plt.title("SPR_BENCH: Validation URA across Epochs\nDifferent batch sizes")
        plt.legend(fontsize=6)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_URA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating URA plot: {e}")
        plt.close()

    # 3. Training Loss
    try:
        plt.figure()
        for bs, tl in zip(batch_sizes, train_loss):
            plt.plot(epochs, tl, label=f"bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training Loss across Epochs\nPer batch size")
        plt.legend(fontsize=6)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_train_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 4. Final Test Accuracy
    try:
        plt.figure()
        plt.bar([str(bs) for bs in batch_sizes], test_acc, color="skyblue")
        plt.xlabel("Batch Size")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Final Test Accuracy\nPer batch size")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar: {e}")
        plt.close()

    # 5. Final Test URA
    try:
        plt.figure()
        plt.bar([str(bs) for bs in batch_sizes], test_ura, color="salmon")
        plt.xlabel("Batch Size")
        plt.ylabel("URA")
        plt.title("SPR_BENCH: Final Test URA\nPer batch size")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_URA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test URA bar: {e}")
        plt.close()
