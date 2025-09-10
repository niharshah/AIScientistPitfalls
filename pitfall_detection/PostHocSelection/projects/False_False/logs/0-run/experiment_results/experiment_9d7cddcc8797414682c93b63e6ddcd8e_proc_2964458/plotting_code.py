import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------- #
#        LOAD EXPERIMENT        #
# ----------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    data_dict = experiment_data["batch_size"]["SPR_BENCH"]
    batch_sizes = sorted(map(int, data_dict.keys()))

    # ----------------------------- #
    #   EXTRACT ARRAYS TO PLOT      #
    # ----------------------------- #
    losses_train, losses_val = {}, {}
    f1_train, f1_val = {}, {}
    final_val_f1, final_swa, final_cwa = [], [], []

    for bs in batch_sizes:
        d = data_dict[str(bs)]
        losses_train[bs] = d["losses"]["train"]
        losses_val[bs] = d["losses"]["val"]
        f1_train[bs] = d["metrics"]["train_macroF1"]
        f1_val[bs] = d["metrics"]["val_macroF1"]
        final_val_f1.append(f1_val[bs][-1])
        final_swa.append(d["SWA"])
        final_cwa.append(d["CWA"])

    # ----------------------------- #
    #           PLOTS               #
    # ----------------------------- #
    # 1) Loss curves
    try:
        plt.figure()
        for bs in batch_sizes:
            epochs = np.arange(1, len(losses_train[bs]) + 1)
            plt.plot(epochs, losses_train[bs], label=f"Train bs={bs}", linestyle="-")
            plt.plot(epochs, losses_val[bs], label=f"Val bs={bs}", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2) Macro-F1 curves
    try:
        plt.figure()
        for bs in batch_sizes:
            epochs = np.arange(1, len(f1_train[bs]) + 1)
            plt.plot(epochs, f1_train[bs], label=f"Train bs={bs}", linestyle="-")
            plt.plot(epochs, f1_val[bs], label=f"Val bs={bs}", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # 3) Final validation Macro-F1 bar chart
    try:
        plt.figure()
        plt.bar(range(len(batch_sizes)), final_val_f1, tick_label=batch_sizes)
        plt.xlabel("Batch Size")
        plt.ylabel("Final Val Macro-F1")
        plt.title("SPR_BENCH: Final Validation Macro-F1 by Batch Size")
        save_path = os.path.join(working_dir, "SPR_BENCH_final_val_macroF1.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating final val F1 bar chart: {e}")
        plt.close()

    # 4) SWA & CWA grouped bars
    try:
        plt.figure()
        idx = np.arange(len(batch_sizes))
        width = 0.35
        plt.bar(idx - width / 2, final_swa, width=width, label="SWA")
        plt.bar(idx + width / 2, final_cwa, width=width, label="CWA")
        plt.xlabel("Batch Size")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH: Shape & Color Weighted Accuracy")
        plt.xticks(idx, batch_sizes)
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_SWA_CWA.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA/CWA bar chart: {e}")
        plt.close()

    # ----------------------------- #
    #     PRINT FINAL METRICS       #
    # ----------------------------- #
    print("\nFinal Validation Metrics")
    print("BatchSize | Macro-F1 |   SWA   |   CWA")
    for i, bs in enumerate(batch_sizes):
        print(
            f"{bs:9d} | {final_val_f1[i]:7.4f} | {final_swa[i]:7.4f} | {final_cwa[i]:7.4f}"
        )
