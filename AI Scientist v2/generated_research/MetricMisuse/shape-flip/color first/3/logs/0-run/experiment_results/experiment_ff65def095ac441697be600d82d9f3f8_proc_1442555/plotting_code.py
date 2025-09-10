import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    spr_exp = experiment_data["dropout_rate"]["SPR_BENCH"]
    best_rate = spr_exp["best_rate"]
    best_dict = spr_exp[f"dr_{best_rate:.2f}"]
    epochs = np.arange(1, len(best_dict["metrics"]["train"]) + 1)

    # ---------------------------------------------------------------
    # 1. BWA curve for best model
    try:
        plt.figure()
        plt.plot(epochs, best_dict["metrics"]["train"], label="Train BWA")
        plt.plot(epochs, best_dict["metrics"]["val"], label="Dev BWA")
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Weighted Accuracy")
        plt.title(f"SPR_BENCH BWA vs Epochs (Best Dropout={best_rate:.2f})")
        plt.legend()
        path = os.path.join(working_dir, f"spr_bwa_curve_best_dr_{best_rate:.2f}.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating BWA curve: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # 2. Loss curve for best model
    try:
        plt.figure()
        plt.plot(epochs, best_dict["losses"]["train"], label="Train Loss")
        plt.plot(epochs, best_dict["losses"]["val"], label="Dev Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Loss vs Epochs (Best Dropout={best_rate:.2f})")
        plt.legend()
        path = os.path.join(working_dir, f"spr_loss_curve_best_dr_{best_rate:.2f}.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating Loss curve: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # 3. Dev BWA across dropout rates
    try:
        rates, val_bwa = [], []
        for k, v in spr_exp.items():
            if k.startswith("dr_"):
                rates.append(float(k.split("_")[1]))
                val_bwa.append(v["metrics"]["val"][-1])  # last epoch val BWA
        order = np.argsort(rates)
        rates = np.array(rates)[order]
        val_bwa = np.array(val_bwa)[order]

        plt.figure()
        plt.bar([f"{r:.2f}" for r in rates], val_bwa, color="skyblue")
        plt.xlabel("Dropout Rate")
        plt.ylabel("Final Dev BWA")
        plt.title("SPR_BENCH Dev BWA by Dropout Rate")
        path = os.path.join(working_dir, "spr_dev_bwa_by_dropout.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating dropout comparison plot: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # 4. Test metrics bar chart
    try:
        test_res = spr_exp["test"]
        metrics = ["BWA", "CWA", "SWA"]
        values = [test_res["BWA"], test_res["CWA"], test_res["SWA"]]
        plt.figure()
        plt.bar(metrics, values, color=["green", "orange", "purple"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Test Metrics (Best Dropout)")
        path = os.path.join(working_dir, "spr_test_metrics_best_dropout.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()
