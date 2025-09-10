import matplotlib.pyplot as plt
import numpy as np
import os

# -------- basic setup ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------- iterate & plot ------
for model_name, model_dict in experiment_data.items():
    for dataset_name, rec in model_dict.items():
        epochs = np.array(rec.get("epochs", []))
        tr_loss = np.array(rec.get("losses", {}).get("train", []))
        vl_loss = np.array(rec.get("losses", {}).get("val", []))
        tr_f1 = np.array(rec.get("metrics", {}).get("train_f1", []))
        vl_f1 = np.array(rec.get("metrics", {}).get("val_f1", []))
        test_f1 = rec.get("metrics", {}).get("test_f1", None)
        sga = rec.get("metrics", {}).get("SGA", None)
        preds = np.array(rec.get("predictions", []))
        gts = np.array(rec.get("ground_truth", []))
        num_classes = len(set(gts)) if gts.size else 0

        # 1) Loss curve
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, vl_loss, label="Validation")
            plt.title(f"{dataset_name} Loss vs Epochs ({model_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy")
            plt.legend()
            fname = f"{dataset_name}_{model_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # 2) F1 curve
        try:
            plt.figure()
            plt.plot(epochs, tr_f1, label="Train")
            plt.plot(epochs, vl_f1, label="Validation")
            plt.title(f"{dataset_name} Macro-F1 vs Epochs ({model_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            fname = f"{dataset_name}_{model_name}_f1_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating F1 curve: {e}")
            plt.close()

        # 3) Bar chart of final metrics
        try:
            plt.figure()
            bars = ["Train_F1", "Val_F1", "Test_F1", "SGA"]
            vals = [
                tr_f1[-1] if tr_f1.size else 0,
                vl_f1[-1] if vl_f1.size else 0,
                test_f1 if test_f1 is not None else 0,
                sga if sga is not None else 0,
            ]
            plt.bar(bars, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
            plt.ylim(0, 1)
            plt.title(f"{dataset_name} Final Metrics ({model_name})")
            fname = f"{dataset_name}_{model_name}_final_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating metrics bar chart: {e}")
            plt.close()

        # 4) Confusion matrix
        if preds.size and gts.size and num_classes <= 50:  # sensible display cap
            try:
                cm = np.zeros((num_classes, num_classes), dtype=int)
                for p, t in zip(preds, gts):
                    cm[t, p] += 1
                plt.figure(figsize=(6, 5))
                plt.imshow(cm, cmap="Blues")
                plt.title(f"{dataset_name} Confusion Matrix ({model_name})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.colorbar()
                plt.tight_layout()
                fname = f"{dataset_name}_{model_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating confusion matrix: {e}")
                plt.close()

        # -------- console summary ----------
        print(f"[{model_name} | {dataset_name}] Test_F1={test_f1:.4f}  SGA={sga:.4f}")
