import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
#                      SETUP & LOAD DATA                             #
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------ #
#                        VISUALISATIONS                              #
# ------------------------------------------------------------------ #
for model_name, dsets in experiment_data.items():
    for dset_name, d in dsets.items():
        t_loss = d["losses"].get("train", [])
        v_loss = d["losses"].get("val", [])
        t_acc = d["metrics"].get("train_acc", [])
        v_acc = d["metrics"].get("val_acc", [])
        fidelity = d["metrics"].get("Rule_Fidelity", [])
        preds = np.asarray(d.get("predictions", []))
        gts = np.asarray(d.get("ground_truth", []))

        # -------- Loss curves -------- #
        try:
            plt.figure()
            plt.plot(t_loss, label="train")
            plt.plot(v_loss, label="val")
            plt.xlabel("epoch")
            plt.ylabel("cross-entropy loss")
            plt.title(f"{dset_name} - Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # -------- Accuracy curves -------- #
        try:
            plt.figure()
            plt.plot(t_acc, label="train")
            plt.plot(v_acc, label="val")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.title(f"{dset_name} - Training vs Validation Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_accuracy_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy curve: {e}")
            plt.close()

        # -------- Rule fidelity -------- #
        try:
            plt.figure()
            plt.plot(fidelity, color="purple")
            plt.xlabel("epoch")
            plt.ylabel("rule fidelity")
            plt.title(f"{dset_name} - Rule Fidelity Over Epochs")
            fname = os.path.join(working_dir, f"{dset_name}_rule_fidelity.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating fidelity plot: {e}")
            plt.close()

        # -------- Confusion matrix -------- #
        try:
            if preds.size and gts.size:
                n_cls = int(max(preds.max(), gts.max()) + 1)
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for p, g in zip(preds, gts):
                    cm[g, p] += 1
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{dset_name} - Confusion Matrix")
                fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

        # ------------------------------------------------------------------ #
        #                    PRINT EVALUATION METRICS                        #
        # ------------------------------------------------------------------ #
        if v_acc:  # last validation accuracy
            print(f"{model_name}/{dset_name} - Final Val Acc: {v_acc[-1]:.4f}")
        if preds.size and gts.size:
            test_acc = (preds == gts).mean()
            print(f"{model_name}/{dset_name} - Test Acc: {test_acc:.4f}")
