import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for ablation, dsets in experiment_data.items():
    for dset_name, dset_blob in dsets.items():
        results = dset_blob.get("results", {})
        # ----------------- 1. Test accuracy vs nhead ---------------- #
        try:
            nheads, test_accs = [], []
            for nhead, blob in results.items():
                nheads.append(int(nhead))
                test_accs.append(blob.get("test_acc", np.nan))
            if nheads:
                idx = np.argsort(nheads)
                nheads = np.array(nheads)[idx]
                test_accs = np.array(test_accs)[idx]
                plt.figure()
                plt.plot(nheads, test_accs, marker="o")
                plt.xlabel("nhead")
                plt.ylabel("Test Accuracy")
                plt.title(f"{dset_name} | {ablation} | Test Accuracy vs nhead")
                save_path = os.path.join(
                    working_dir, f"{dset_name}_{ablation}_test_acc_vs_nhead.png"
                )
                plt.savefig(save_path)
                plt.close()
        except Exception as e:
            print(f"Error plotting test-accuracy curve: {e}")
            plt.close()

        # ----------- 2. Per-head training / validation curves ------- #
        for i, (nhead, blob) in enumerate(results.items()):
            metrics = blob.get("metrics", {})
            losses = blob.get("losses", {})
            epochs = range(1, 1 + len(metrics.get("train_acc", [])))

            # Accuracy plot
            try:
                plt.figure()
                plt.plot(epochs, metrics.get("train_acc", []), label="train")
                plt.plot(epochs, metrics.get("val_acc", []), label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(
                    f"{dset_name} | {ablation} | nhead={nhead}\nTraining vs Validation Accuracy"
                )
                plt.legend()
                fname = f"{dset_name}_{ablation}_nhead{nhead}_acc_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating acc plot for nhead={nhead}: {e}")
                plt.close()

            # Loss plot
            try:
                plt.figure()
                plt.plot(epochs, losses.get("train_loss", []), label="train")
                plt.plot(epochs, losses.get("val_loss", []), label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(
                    f"{dset_name} | {ablation} | nhead={nhead}\nTraining vs Validation Loss"
                )
                plt.legend()
                fname = f"{dset_name}_{ablation}_nhead{nhead}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating loss plot for nhead={nhead}: {e}")
                plt.close()

            # Confusion matrix (limit to first 4 heads)
            if i < 4:
                try:
                    preds = np.array(blob.get("predictions", []), dtype=int)
                    gts = np.array(blob.get("ground_truth", []), dtype=int)
                    if preds.size and gts.size:
                        num_classes = max(preds.max(), gts.max()) + 1
                        cm = np.zeros((num_classes, num_classes), dtype=int)
                        for gt, pr in zip(gts, preds):
                            cm[gt, pr] += 1
                        plt.figure()
                        plt.imshow(cm, cmap="Blues")
                        plt.colorbar()
                        plt.xlabel("Predicted")
                        plt.ylabel("Ground Truth")
                        plt.title(
                            f"{dset_name} | {ablation} | nhead={nhead}\nConfusion Matrix"
                        )
                        fname = (
                            f"{dset_name}_{ablation}_nhead{nhead}_confusion_matrix.png"
                        )
                        plt.savefig(os.path.join(working_dir, fname))
                        plt.close()
                except Exception as e:
                    print(f"Error creating confusion matrix for nhead={nhead}: {e}")
                    plt.close()
