import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch nested dict safely
def get(path, default=None):
    d = experiment_data
    for p in path:
        d = d.get(p, {})
    return d or default


data_root = experiment_data.get("uni_directional_encoder", {}).get("SPR", {})
if not data_root:
    print("No SPR data found.")
    exit()

# unpack series --------------------------------------------------------------
contr = np.array(data_root["contrastive_losses"])  # (ep, loss)
tr_loss = np.array(data_root["losses"]["train"])  # (ep, loss)
val_loss = np.array(data_root["losses"]["val"])
SWA = np.array(data_root["metrics"]["SWA"])
CWA = np.array(data_root["metrics"]["CWA"])
Comp = np.array(data_root["metrics"]["CompWA"])

# ------------------ fig 1: contrastive loss
try:
    plt.figure()
    if contr.size:
        plt.plot(contr[:, 0], contr[:, 1], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        plt.title("Contrastive Pre-training Loss\nDataset: SPR")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No data", ha="center")
    fname = os.path.join(working_dir, "SPR_contrastive_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating contrastive plot: {e}")
    plt.close()

# ------------------ fig 2: train & val loss
try:
    plt.figure()
    if tr_loss.size:
        plt.plot(tr_loss[:, 0], tr_loss[:, 1], label="Train")
    if val_loss.size:
        plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Fine-tuning Loss Curves\nDataset: SPR (Left: Train, Right: Val)")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(working_dir, "SPR_train_val_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------ fig 3: metrics
try:
    plt.figure()
    if SWA.size:
        plt.plot(SWA[:, 0], SWA[:, 1], label="SWA")
    if CWA.size:
        plt.plot(CWA[:, 0], CWA[:, 1], label="CWA")
    if Comp.size:
        plt.plot(Comp[:, 0], Comp[:, 1], label="CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("Evaluation Metrics over Epochs\nDataset: SPR")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(working_dir, "SPR_weighted_accuracy_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ------------------ print final metrics
if Comp.size:
    ep, swa = SWA[-1]
    _, cwa = CWA[-1]
    _, cwa2 = Comp[-1]
    print(
        f"Final epoch ({int(ep)}) metrics - SWA: {swa:.4f}, CWA: {cwa:.4f}, CompWA: {cwa2:.4f}"
    )
