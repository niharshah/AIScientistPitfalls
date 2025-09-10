import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
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
    data = experiment_data.get("dual_channel", {})
    losses = data.get("losses", {})
    metrics_val = data.get("metrics", {}).get("val", [])
    preds, gts = data.get("predictions", []), data.get("ground_truth", [])

    # ---------- 1: loss curve ---------------------------------
    try:
        tr_epochs, tr_losses = zip(*losses.get("train", []))
        val_epochs, val_losses = zip(*losses.get("val", []))

        plt.figure()
        plt.plot(tr_epochs, tr_losses, label="Train")
        plt.plot(val_epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve\nLeft: Train, Right: Val")
        plt.legend()
        fname = "dual_channel_loss_curve_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curve: {e}")
        plt.close()

    # ---------- 2: metric curves ------------------------------
    try:
        ep, cwa, swa, pcwa = [], [], [], []
        for t in metrics_val:
            ep.append(t[0])
            cwa.append(t[1]["CWA"])
            swa.append(t[1]["SWA"])
            pcwa.append(t[1]["PCWA"])

        plt.figure()
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, pcwa, label="PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Validation Metrics Over Epochs")
        plt.legend()
        fname = "dual_channel_metric_curves_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting metric curves: {e}")
        plt.close()

    # ---------- 3: final metric bar chart ---------------------
    try:
        last_dict = metrics_val[-1][1] if metrics_val else {}
        names = ["CWA", "SWA", "PCWA"]
        vals = [last_dict.get(k, 0) for k in names]

        plt.figure()
        plt.bar(names, vals, color=["steelblue", "salmon", "seagreen"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Validation Metrics")
        fname = "dual_channel_final_val_metrics_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting final metric bar chart: {e}")
        plt.close()

    # ---------- 4: confusion matrix heatmap -------------------
    try:
        if preds and gts:
            labels = sorted(set(gts) | set(preds))
            idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[idx[t], idx[p]] += 1

            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            fname = "dual_channel_confusion_matrix_SPR_BENCH.png"
            plt.savefig(os.path.join(working_dir, fname), bbox_inches="tight")
            plt.close()
        else:
            print("Predictions / Ground truth missing, skipping confusion matrix.")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        plt.close()

    # ---------- print summary ---------------------------------
    if metrics_val:
        print(f"Final Val Metrics: {last_dict}")
    if preds and gts:
        acc = sum(int(a == b) for a, b in zip(preds, gts)) / len(gts)
        print(f"Test Accuracy: {acc:.4f}")
else:
    print("No experiment data to visualize.")
