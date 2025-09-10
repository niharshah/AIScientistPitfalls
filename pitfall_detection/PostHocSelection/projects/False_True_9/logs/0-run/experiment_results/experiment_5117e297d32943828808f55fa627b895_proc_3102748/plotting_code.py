import matplotlib.pyplot as plt
import numpy as np
import os

# --------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data -------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dname, rec in experiment_data.items():
    # -------- extract ----------
    losses = rec.get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    metrics = rec.get("metrics", {}).get("val", [])  # (ep,swa,cwa,dawa)
    preds = np.array(rec.get("predictions", []))
    gts = np.array(rec.get("ground_truth", []))

    # ---------- plot 1: loss curves ----------
    try:
        plt.figure()
        if train_loss:
            ep_t, v_t = zip(*train_loss)
            plt.plot(ep_t, v_t, label="train")
        if val_loss:
            ep_v, v_v = zip(*val_loss)
            plt.plot(ep_v, v_v, "--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dname}: {e}")
        plt.close()

    # ---------- plot 2: metric curves ----------
    try:
        if metrics:
            ep, swa, cwa, dawa = zip(*metrics)
            plt.figure()
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, dawa, label="DAWA")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname}: Validation Metrics per Epoch")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_metric_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating metric curves for {dname}: {e}")
        plt.close()

    # ---------- plot 3: final metric bar ----------
    try:
        if metrics:
            final_swa, final_cwa, final_dawa = swa[-1], cwa[-1], dawa[-1]
            plt.figure()
            plt.bar(
                ["SWA", "CWA", "DAWA"],
                [final_swa, final_cwa, final_dawa],
                color=["steelblue", "orange", "green"],
            )
            plt.ylim(0, 1)
            plt.title(f"{dname}: Final-Epoch Accuracy Metrics")
            plt.savefig(os.path.join(working_dir, f"{dname}_final_metrics_bar.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating final metric bar for {dname}: {e}")
        plt.close()

    # ---------- plot 4: confusion matrix ----------
    try:
        if preds.size and gts.size:
            labels = sorted(set(gts) | set(preds))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[labels.index(t), labels.index(p)] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dname}: Confusion Matrix (dev set)")
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # ---------- print summary ----------
    if metrics:
        print(
            f"{dname} final metrics -> SWA:{swa[-1]:.4f}  CWA:{cwa[-1]:.4f}  DAWA:{dawa[-1]:.4f}"
        )
