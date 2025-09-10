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
    bench = experiment_data["batch_size"]["SPR_BENCH"]
    bs_list = sorted(bench.keys())
    epochs_dict = {bs: bench[bs]["epochs"] for bs in bs_list}

    # 1) Training loss comparison
    try:
        plt.figure()
        for bs in bs_list:
            plt.plot(
                epochs_dict[bs],
                bench[bs]["losses"]["train"],
                marker="o",
                label=f"bs={bs}",
            )
        plt.title(
            "SPR_BENCH – Training Loss vs Epochs\nLeft: Smaller bs, Right: Larger bs"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_comparison.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss comparison: {e}")
        plt.close()

    # 2) Validation CpxWA across batch sizes
    try:
        plt.figure()
        for bs in bs_list:
            cpx = [m["cpx"] for m in bench[bs]["metrics"]["val"]]
            plt.plot(epochs_dict[bs], cpx, marker="o", label=f"bs={bs}")
        plt.title(
            "SPR_BENCH – Validation Complexity-Weighted Accuracy\nHigher is better"
        )
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_cpxwa_comparison.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating CpxWA comparison: {e}")
        plt.close()

    # choose one batch size for detailed train/val plots
    focus_bs = 32 if 32 in bs_list else bs_list[0]
    focus_epochs = epochs_dict[focus_bs]
    tr_metrics = bench[focus_bs]["metrics"]["train"]
    val_metrics = bench[focus_bs]["metrics"]["val"]

    # helper to extract metric series
    def series(key, split):
        return [m[key] for m in (tr_metrics if split == "train" else val_metrics)]

    # 3) Train vs Val CpxWA
    try:
        plt.figure()
        plt.plot(focus_epochs, series("cpx", "train"), marker="o", label="Train")
        plt.plot(focus_epochs, series("cpx", "val"), marker="s", label="Validation")
        plt.title(f"SPR_BENCH – Train vs Val CpxWA (bs={focus_bs})")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_bs{focus_bs}_cpxwa_train_val.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating CpxWA train/val: {e}")
        plt.close()

    # 4) Train vs Val SWA
    try:
        plt.figure()
        plt.plot(focus_epochs, series("swa", "train"), marker="o", label="Train")
        plt.plot(focus_epochs, series("swa", "val"), marker="s", label="Validation")
        plt.title(f"SPR_BENCH – Train vs Val Shape-Weighted Acc (bs={focus_bs})")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_bs{focus_bs}_swa_train_val.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA train/val: {e}")
        plt.close()

    # 5) Train vs Val CWA
    try:
        plt.figure()
        plt.plot(focus_epochs, series("cwa", "train"), marker="o", label="Train")
        plt.plot(focus_epochs, series("cwa", "val"), marker="s", label="Validation")
        plt.title(f"SPR_BENCH – Train vs Val Color-Weighted Acc (bs={focus_bs})")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_bs{focus_bs}_cwa_train_val.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating CWA train/val: {e}")
        plt.close()
