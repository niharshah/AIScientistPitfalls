import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------- load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
models = list(experiment_data.get(dataset, {}).keys())
if not models:
    print("No data found, aborting plots.")
else:
    epoch_nums = range(
        1, len(experiment_data[dataset][models[0]]["losses"]["train"]) + 1
    )

    # -------------------------------------------------------- 1) loss curves
    try:
        plt.figure()
        for m in models:
            tr = [d["loss"] for d in experiment_data[dataset][m]["losses"]["train"]]
            val = [d["loss"] for d in experiment_data[dataset][m]["losses"]["val"]]
            plt.plot(epoch_nums, tr, label=f"{m}-train")
            plt.plot(epoch_nums, val, "--", label=f"{m}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------------------------------------------------------- 2) macro-F1 curves
    try:
        plt.figure()
        for m in models:
            tr = [
                d["macro_f1"] for d in experiment_data[dataset][m]["metrics"]["train"]
            ]
            val = [d["macro_f1"] for d in experiment_data[dataset][m]["metrics"]["val"]]
            plt.plot(epoch_nums, tr, label=f"{m}-train")
            plt.plot(epoch_nums, val, "--", label=f"{m}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating macro-F1 plot: {e}")
        plt.close()

    # -------------------------------------------------------- 3) RGA / accuracy curves
    try:
        plt.figure()
        for m in models:
            val = [d["RGA"] for d in experiment_data[dataset][m]["metrics"]["val"]]
            plt.plot(epoch_nums, val, label=f"{m}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (RGA)")
        plt.title("SPR_BENCH: Validation Accuracy over Epochs")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # -------------------------------------------------------- 4) final-epoch bar chart
    try:
        last_f1 = [
            experiment_data[dataset][m]["metrics"]["val"][-1]["macro_f1"]
            for m in models
        ]
        last_acc = [
            experiment_data[dataset][m]["metrics"]["val"][-1]["RGA"] for m in models
        ]
        x = np.arange(len(models))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, last_f1, width, label="Macro-F1")
        plt.bar(x + width / 2, last_acc, width, label="Accuracy")
        plt.xticks(x, models)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Final Validation Scores")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_final_scores.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final score bar chart: {e}")
        plt.close()

    # -------------------------------------------------------- print final metrics
    print("\nFinal Validation Metrics:")
    for m in models:
        f1 = experiment_data[dataset][m]["metrics"]["val"][-1]["macro_f1"]
        acc = experiment_data[dataset][m]["metrics"]["val"][-1]["RGA"]
        print(f"{m}: Macro-F1={f1:.3f}, Accuracy={acc:.3f}")
