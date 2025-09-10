import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# helper to fetch data safely
def load_data():
    try:
        return np.load(
            os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
        ).item()
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return None


exp = load_data()
if exp is None:
    quit()

# extract the only run we expect
try:
    run = exp["TokenOnlyTransformer"]["SPR_BENCH"]
    epochs = np.array(run.get("epochs", []))
    losses_tr = np.array(run["losses"]["train"])
    losses_val = np.array(run["losses"]["val"])
    metrics_tr = run["metrics"]["train"]
    metrics_val = run["metrics"]["val"]
    acc_tr = np.array([m["acc"] for m in metrics_tr])
    acc_val = np.array([m["acc"] for m in metrics_val])
    mcc_tr = np.array([m["MCC"] for m in metrics_tr])
    mcc_val = np.array([m["MCC"] for m in metrics_val])
    rma_tr = np.array([m["RMA"] for m in metrics_tr])
    rma_val = np.array([m["RMA"] for m in metrics_val])
except Exception as e:
    print(f"Error extracting run data: {e}")
    quit()

plots = [
    ("loss", losses_tr, losses_val, "Loss"),
    ("accuracy", acc_tr, acc_val, "Accuracy"),
    ("mcc", mcc_tr, mcc_val, "Matthews Corr. Coef."),
    ("rma", rma_tr, rma_val, "Rule Macro Accuracy"),
]

# iterate and create up to 5 figures
for name, tr, val, label in plots[:5]:
    try:
        plt.figure()
        plt.plot(epochs, tr, label="Train")
        plt.plot(epochs, val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"SPR_BENCH – Train vs Val {label}")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{name}_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {name} plot: {e}")
        plt.close()
