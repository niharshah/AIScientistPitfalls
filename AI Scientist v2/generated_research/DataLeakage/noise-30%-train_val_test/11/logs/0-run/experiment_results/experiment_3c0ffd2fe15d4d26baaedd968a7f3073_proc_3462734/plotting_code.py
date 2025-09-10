import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------------------------------------------------------------
summary = []  # collect (bs, best_val_f1, test_f1)

for dataset_name, bs_dict in experiment_data.get("batch_size_tuning", {}).items():
    for bs, res in bs_dict.items():
        epochs = res.get("epochs", [])
        tr_loss = res["losses"]["train"]
        val_loss = res["losses"]["val"]
        tr_f1 = res["metrics"]["train_macro_f1"]
        val_f1 = res["metrics"]["val_macro_f1"]
        test_f1 = res["metrics"]["test_macro_f1"]
        best_val_f1 = max(val_f1) if val_f1 else None
        summary.append((int(bs), best_val_f1, test_f1))

        # -------- Loss curve ---------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dataset_name} | Loss Curves | Batch Size {bs}")
            plt.legend()
            fname = f"{dataset_name}_loss_curves_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for bs={bs}: {e}")
            plt.close()

        # -------- F1 curve -----------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, tr_f1, label="Train Macro-F1")
            plt.plot(epochs, val_f1, label="Validation Macro-F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dataset_name} | Macro-F1 Curves | Batch Size {bs}")
            plt.legend()
            fname = f"{dataset_name}_f1_curves_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating F1 plot for bs={bs}: {e}")
            plt.close()

# -------- Comparative bar chart of test F1 ------------------------------------
try:
    if summary:
        summary = sorted(summary, key=lambda x: x[0])  # sort by batch size
        bss, _, test_f1s = zip(*summary)
        plt.figure()
        plt.bar(range(len(bss)), test_f1s, tick_label=bss)
        plt.xlabel("Batch Size")
        plt.ylabel("Test Macro-F1")
        plt.title("SPR_BENCH | Test Macro-F1 vs Batch Size")
        fname = "SPR_BENCH_test_f1_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating comparative bar chart: {e}")
    plt.close()

# -------- Print summary --------------------------------------------------------
print("Batch Size | Best Val F1 | Test F1")
for bs, best_val, test in summary:
    print(f"{bs:10d} | {best_val:11.4f} | {test:.4f}")
