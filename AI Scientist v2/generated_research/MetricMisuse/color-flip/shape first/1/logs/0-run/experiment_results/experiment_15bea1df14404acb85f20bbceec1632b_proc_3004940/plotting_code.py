import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- plotting ----------
for dname, dct in experiment_data.items():
    # ----- 1: pre-training loss -----
    try:
        pts = dct["losses"].get("pretrain", [])
        if pts:
            plt.figure()
            plt.plot(range(1, len(pts) + 1), pts, marker="o")
            plt.xlabel("Pre-training epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Pre-training Loss Curve")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_pretrain_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating pretrain plot: {e}")
        plt.close()

    # ----- 2: fine-tune loss -----
    try:
        tr = dct["losses"].get("train", [])
        vl = dct["losses"].get("val", [])
        if tr or vl:
            plt.figure()
            if tr:
                plt.plot(range(1, len(tr) + 1), tr, label="train", marker="o")
            if vl:
                plt.plot(range(1, len(vl) + 1), vl, label="val", marker="o")
            plt.xlabel("Fine-tune epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Training / Validation Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_ft_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating FT loss plot: {e}")
        plt.close()

    # ----- 3: metric curves -----
    try:
        ep = dct.get("epochs", [])
        swa = dct["metrics"].get("SWA", [])
        cwa = dct["metrics"].get("CWA", [])
        scwa = dct["metrics"].get("SCWA", [])
        if ep and (swa or cwa or scwa):
            plt.figure()
            if swa:
                plt.plot(ep, swa, label="SWA", marker="o")
            if cwa:
                plt.plot(ep, cwa, label="CWA", marker="o")
            if scwa:
                plt.plot(ep, scwa, label="SCWA", marker="o")
            plt.xlabel("Fine-tune epoch")
            plt.ylabel("Weighted accuracy")
            plt.title(f"{dname} – Metric Curves")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # ----- 4: confusion matrix -----
    try:
        y_true = np.array(dct.get("ground_truth", []))
        y_pred = np.array(dct.get("predictions", []))
        if y_true.size and y_pred.size:
            labels = np.unique(np.concatenate([y_true, y_pred]))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.title(f"{dname} – Confusion Matrix")
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

print("Finished generating plots.")
