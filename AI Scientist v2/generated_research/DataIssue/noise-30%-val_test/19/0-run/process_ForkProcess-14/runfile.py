import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & data loading ----------
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
    logs = experiment_data.get("transformer", {})
    epochs_info = logs.get("epochs", [])
    tr_loss = logs.get("losses", {}).get("train", [])
    val_loss = logs.get("losses", {}).get("val", [])
    tr_mcc = logs.get("metrics", {}).get("train_MCC", [])
    val_mcc = logs.get("metrics", {}).get("val_MCC", [])

    # ------------- organise by dropout -------------
    by_dp = {}
    for i, (dp, ep) in enumerate(epochs_info):
        d = by_dp.setdefault(
            dp,
            {"epoch": [], "tr_loss": [], "val_loss": [], "tr_mcc": [], "val_mcc": []},
        )
        d["epoch"].append(ep)
        d["tr_loss"].append(tr_loss[i] if i < len(tr_loss) else np.nan)
        d["val_loss"].append(val_loss[i] if i < len(val_loss) else np.nan)
        d["tr_mcc"].append(tr_mcc[i] if i < len(tr_mcc) else np.nan)
        d["val_mcc"].append(val_mcc[i] if i < len(val_mcc) else np.nan)

    # -------------------- 1. loss curves --------------------
    try:
        plt.figure(figsize=(6, 4))
        for dp, d in by_dp.items():
            plt.plot(d["epoch"], d["tr_loss"], label=f"Train dp={dp}")
            plt.plot(d["epoch"], d["val_loss"], linestyle="--", label=f"Val dp={dp}")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Loss Curves — synthetic SPR_BENCH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves: {e}")
        plt.close()

    # -------------------- 2. MCC curves --------------------
    try:
        plt.figure(figsize=(6, 4))
        for dp, d in by_dp.items():
            plt.plot(d["epoch"], d["tr_mcc"], label=f"Train dp={dp}")
            plt.plot(d["epoch"], d["val_mcc"], linestyle="--", label=f"Val dp={dp}")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("MCC Curves — synthetic SPR_BENCH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_mcc_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting MCC curves: {e}")
        plt.close()

    # -------------------- 3. bar chart dev MCC --------------------
    try:
        dps, final_dev = [], []
        for dp, d in by_dp.items():
            if d["val_mcc"]:
                dps.append(str(dp))
                final_dev.append(d["val_mcc"][-1])
        plt.figure(figsize=(5, 4))
        plt.bar(dps, final_dev, color="steelblue")
        plt.xlabel("Dropout")
        plt.ylabel("Final Dev MCC")
        plt.title("Final Dev MCC by Dropout — synthetic SPR_BENCH")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_dev_mcc_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting dev MCC bar chart: {e}")
        plt.close()

    # -------------------- 4. confusion matrix --------------------
    try:
        y_pred = np.array(logs.get("predictions", []))
        y_true = np.array(logs.get("ground_truth", []))
        if y_pred.size and y_true.size:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            cm = np.array([[tn, fp], [fn, tp]])
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.title("Confusion Matrix — synthetic SPR_BENCH")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        plt.close()

    # -------------------- 5. test metric bars --------------------
    try:
        test_mcc = logs.get("test_MCC")
        test_f1 = logs.get("test_F1")
        if test_mcc is not None and test_f1 is not None:
            metrics = ["MCC", "Macro-F1"]
            scores = [test_mcc, test_f1]
            plt.figure(figsize=(4, 4))
            plt.bar(metrics, scores, color=["salmon", "seagreen"])
            plt.ylim(0, 1)
            plt.title("Test Metrics — synthetic SPR_BENCH")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "spr_bench_test_metrics.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting test metric bars: {e}")
        plt.close()
