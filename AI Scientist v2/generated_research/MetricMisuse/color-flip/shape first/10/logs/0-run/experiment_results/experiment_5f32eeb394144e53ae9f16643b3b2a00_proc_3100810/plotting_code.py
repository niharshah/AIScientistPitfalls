import matplotlib.pyplot as plt
import numpy as np
import os

# prepare directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# iterate over datasets
for ds_name, bench in experiment_data.items():
    # basic tensors ------------------------------------------------------------
    tr_loss = np.array(bench["losses"].get("train", []))
    val_loss = np.array(bench["losses"].get("val", []))
    con_loss = np.array(bench["losses"].get("contrastive", []))
    swa = np.array(bench["metrics"].get("SWA_val", []))
    cwa = np.array(bench["metrics"].get("CWA_val", []))
    scwa = np.array(bench["metrics"].get("SCWA_val", []))
    preds = bench.get("predictions", [])
    gts = bench.get("ground_truth", [])

    epochs_cls = np.arange(1, len(tr_loss) + 1)
    epochs_con = np.arange(1, len(con_loss) + 1)

    # 1) classification loss ---------------------------------------------------
    try:
        if tr_loss.size and val_loss.size:
            plt.figure()
            plt.plot(epochs_cls, tr_loss, label="Train")
            plt.plot(epochs_cls, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("CE Loss")
            plt.title(f"{ds_name}: Training vs. Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name.lower()}_loss_curves.png")
            plt.savefig(fname)
            print("Saved", fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) contrastive loss ------------------------------------------------------
    try:
        if con_loss.size:
            plt.figure()
            plt.plot(epochs_con, con_loss, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title(f"{ds_name}: Contrastive Pre-training Loss")
            fname = os.path.join(working_dir, f"{ds_name.lower()}_contrastive_loss.png")
            plt.savefig(fname)
            print("Saved", fname)
            plt.close()
    except Exception as e:
        print(f"Error creating contrastive plot for {ds_name}: {e}")
        plt.close()

    # 3) metric curves ---------------------------------------------------------
    try:
        if swa.size:
            plt.figure()
            plt.plot(epochs_cls, swa, label="SWA")
            plt.plot(epochs_cls, cwa, label="CWA")
            plt.plot(epochs_cls, scwa, label="SCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{ds_name}: Weighted Accuracy Metrics")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name.lower()}_metric_curves.png")
            plt.savefig(fname)
            print("Saved", fname)
            plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {ds_name}: {e}")
        plt.close()

    # 4) accuracy per epoch -----------------------------------------------------
    try:
        if preds and gts and len(preds) == len(gts):
            acc = [(np.array(p) == np.array(g)).mean() for p, g in zip(preds, gts)]
            plt.figure()
            plt.plot(np.arange(1, len(acc) + 1), acc, marker="o")
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name}: Accuracy per Epoch")
            fname = os.path.join(working_dir, f"{ds_name.lower()}_accuracy.png")
            plt.savefig(fname)
            print("Saved", fname)
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # print final metrics ------------------------------------------------------
    if scwa.size:
        print(
            f"{ds_name}: Final epoch SCWA={scwa[-1]:.3f}, "
            f"SWA={swa[-1]:.3f}, CWA={cwa[-1]:.3f}"
        )
