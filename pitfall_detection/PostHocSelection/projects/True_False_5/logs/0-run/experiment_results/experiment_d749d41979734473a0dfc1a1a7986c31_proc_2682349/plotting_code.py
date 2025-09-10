import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# convenience helpers ----------------------------------------------------------
def get_exp_record(exp_dict):
    # assumes single model/dataset entry as produced by default script
    try:
        model_key = next(iter(exp_dict))
        dset_key = next(iter(exp_dict[model_key]))
        return exp_dict[model_key][dset_key], dset_key
    except Exception:
        return None, None


exp_rec, dset_name = get_exp_record(experiment_data)
if exp_rec is None:
    print("No experiment data found to plot.")
    exit()

# Plot 1: loss curves ----------------------------------------------------------
try:
    tr_loss = np.asarray(exp_rec["losses"]["train"])
    vl_loss = np.asarray(exp_rec["losses"]["val"])
    if tr_loss.size and vl_loss.size:
        plt.figure()
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, vl_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name} Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
    else:
        print("Loss data empty; skipping loss curve.")
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# Plot 2: SWA curves -----------------------------------------------------------
try:
    tr_swa = np.asarray(exp_rec["metrics"]["train_swa"])
    vl_swa = np.asarray(exp_rec["metrics"]["val_swa"])
    if tr_swa.size and vl_swa.size:
        plt.figure()
        epochs = np.arange(1, len(tr_swa) + 1)
        plt.plot(epochs, tr_swa, label="Train SWA")
        plt.plot(epochs, vl_swa, label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dset_name} Shape-Weighted Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset_name}_swa_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
    else:
        print("SWA data empty; skipping accuracy curve.")
except Exception as e:
    print(f"Error creating SWA curve: {e}")
finally:
    plt.close()

# Plot 3: confusion matrix -----------------------------------------------------
try:
    preds = np.asarray(exp_rec.get("predictions", []), dtype=int)
    gts = np.asarray(exp_rec.get("ground_truth", []), dtype=int)
    if preds.size and gts.size and preds.shape == gts.shape:
        num_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, shrink=0.75)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            f"{dset_name} Confusion Matrix\n(Left axis: True, Bottom axis: Pred.)"
        )
        fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
        plt.savefig(fname)
        print("Saved", fname)
    else:
        print("Prediction/GT data empty; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
finally:
    plt.close()
