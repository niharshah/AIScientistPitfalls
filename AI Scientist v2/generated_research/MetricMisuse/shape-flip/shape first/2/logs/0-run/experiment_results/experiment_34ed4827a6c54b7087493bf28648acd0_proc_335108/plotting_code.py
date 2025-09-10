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
    run = experiment_data["ablation_no_hidden_layer"]["spr_bench"]
    epochs = run.get("epochs", [])
    train_loss = run["losses"].get("train", [])
    dev_loss = run["losses"].get("dev", [])
    train_pha = run["metrics"].get("train_PHA", [])
    dev_pha = run["metrics"].get("dev_PHA", [])
    preds = np.asarray(run.get("predictions", []))
    gts = np.asarray(run.get("ground_truth", []))
    test_metrics = run.get("test_metrics", {})

    # 1. Loss curve
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, dev_loss, label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("spr_bench Loss Curve\nTrain vs Dev")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2. PHA curve
    try:
        plt.figure()
        plt.plot(epochs, train_pha, label="Train PHA")
        plt.plot(epochs, dev_pha, label="Dev PHA")
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title("spr_bench PHA Curve\nTrain vs Dev")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_PHA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating PHA curve: {e}")
        plt.close()

    # 3. Test metric bars
    try:
        if test_metrics:
            plt.figure()
            names = list(test_metrics.keys())
            vals = [test_metrics[k] for k in names]
            plt.bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
            plt.ylim(0, 1)
            for idx, v in enumerate(vals):
                plt.text(idx, v + 0.02, f"{v:.2f}", ha="center")
            plt.title("spr_bench Test Metrics\nSWA / CWA / PHA")
            fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot: {e}")
        plt.close()

    # 4. Confusion matrix
    try:
        if preds.size and gts.size:
            n_cls = int(max(gts.max(), preds.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                "spr_bench Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

print(f"Plots saved to {working_dir}")
