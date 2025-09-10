import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- set up -------------------
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
    for dset_name, logs in experiment_data.items():
        # ---------- helpers ----------
        tr_loss = logs["losses"].get("train", [])
        val_loss = logs["losses"].get("val", [])
        val_metrics = logs["metrics"].get("val", [])
        epochs = list(range(1, len(val_loss) + 1))

        def get_metric(field):
            return [m.get(field) for m in val_metrics] if val_metrics else []

        acc = get_metric("acc")
        hpa = get_metric("HPA")
        cwa = get_metric("CWA")
        swa = get_metric("SWA")
        preds = logs.get("predictions", [])
        gts = logs.get("ground_truth", [])

        # ---------- Fig 1: Loss curves ----------
        try:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
            if tr_loss:
                ax[0].plot(epochs, tr_loss, label="train")
            if val_loss:
                ax[1].plot(epochs, val_loss, label="val", color="tab:orange")
            ax[0].set_title("Left: Train Loss")
            ax[1].set_title("Right: Validation Loss")
            for a in ax:
                a.set_xlabel("Epoch")
                a.set_ylabel("Loss")
                a.legend()
            fig.suptitle(f"{dset_name} Loss Curves")
            fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating loss plot for {dset_name}: {e}")
            plt.close()

        # ---------- Fig 2: Acc & HPA ----------
        try:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
            if acc:
                ax[0].plot(epochs, acc, label="Accuracy")
            if hpa:
                ax[1].plot(epochs, hpa, label="HPA", color="tab:green")
            ax[0].set_title("Left: Accuracy")
            ax[1].set_title("Right: Harmonic Poly Accuracy")
            for a in ax:
                a.set_xlabel("Epoch")
                a.set_ylabel("Score")
                a.legend()
            fig.suptitle(f"{dset_name} Accuracy Metrics")
            fname = os.path.join(working_dir, f"{dset_name}_acc_hpa.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating acc/HPA plot for {dset_name}: {e}")
            plt.close()

        # ---------- Fig 3: CWA & SWA ----------
        try:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
            if cwa:
                ax[0].plot(epochs, cwa, label="CWA")
            if swa:
                ax[1].plot(epochs, swa, label="SWA", color="tab:red")
            ax[0].set_title("Left: Color-Weighted Acc")
            ax[1].set_title("Right: Shape-Weighted Acc")
            for a in ax:
                a.set_xlabel("Epoch")
                a.set_ylabel("Score")
                a.legend()
            fig.suptitle(f"{dset_name} Weighted Accuracies")
            fname = os.path.join(working_dir, f"{dset_name}_cwa_swa.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating CWA/SWA plot for {dset_name}: {e}")
            plt.close()

        # ---------- Fig 4: Prediction distribution ----------
        try:
            if preds and gts:
                labels = sorted(set(gts + preds))
                gt_counts = [gts.count(l) for l in labels]
                pr_counts = [preds.count(l) for l in labels]

                x = np.arange(len(labels))
                width = 0.35
                plt.figure(figsize=(6, 4), dpi=120)
                plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
                plt.bar(x + width / 2, pr_counts, width, label="Predicted")
                plt.xticks(x, labels)
                plt.title(f"{dset_name}: GT vs Predicted Label Counts")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset_name}_label_counts.png")
                plt.savefig(fname)
                plt.close()
                print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating label count plot for {dset_name}: {e}")
            plt.close()
