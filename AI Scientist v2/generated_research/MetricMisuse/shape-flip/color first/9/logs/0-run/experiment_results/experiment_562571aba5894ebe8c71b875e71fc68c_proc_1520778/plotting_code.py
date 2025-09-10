import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data -------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["remove_similarity_edges"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:
    epochs = spr["epochs"]
    loss_tr, loss_val = spr["losses"]["train"], spr["losses"]["val"]
    met_tr = {k: [m[k] for m in spr["metrics"]["train"]] for k in ["HWA", "CWA", "SWA"]}
    met_val = {k: [m[k] for m in spr["metrics"]["val"]] for k in ["HWA", "CWA", "SWA"]}
    preds, gts = spr["predictions"], spr["ground_truth"]

    # -------- helper to make 2-line plots -------- #
    def dual_plot(y1, y2, title, fname, ylabel):
        try:
            plt.figure()
            plt.plot(epochs, y1, label="Train")
            plt.plot(epochs, y2, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"SPR: {title}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname), dpi=150)
        except Exception as e:
            print(f"Error creating {fname}: {e}")
        finally:
            plt.close()

    # 1-4) curves
    dual_plot(loss_tr, loss_val, "Loss Curve", "SPR_loss_curve.png", "Loss")
    dual_plot(met_tr["HWA"], met_val["HWA"], "HWA Curve", "SPR_HWA_curve.png", "HWA")
    dual_plot(met_tr["CWA"], met_val["CWA"], "CWA Curve", "SPR_CWA_curve.png", "CWA")
    dual_plot(met_tr["SWA"], met_val["SWA"], "SWA Curve", "SPR_SWA_curve.png", "SWA")

    # 5) confusion matrix
    try:
        num_cls = max(max(preds), max(gts)) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR: Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"), dpi=150)
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    finally:
        plt.close()
