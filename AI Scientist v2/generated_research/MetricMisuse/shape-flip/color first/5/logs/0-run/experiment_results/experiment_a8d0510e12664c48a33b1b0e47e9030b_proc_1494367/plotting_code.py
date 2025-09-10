import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()["SPR_dataset"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    ep = exp["epochs"]
    losses = exp["losses"]
    mets = exp["metrics"]
    test = exp["test"]

    # 1) Loss curves ----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(ep, losses["train"], label="Train")
        plt.plot(ep, losses["val"], "--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_dataset – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_dataset_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) CWA curves -----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(ep, mets["CWA"]["train"], label="Train")
        plt.plot(ep, mets["CWA"]["val"], "--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.title("SPR_dataset – Training vs Validation CWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_dataset_CWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA curves: {e}")
        plt.close()

    # 3) SWA curves -----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(ep, mets["SWA"]["train"], label="Train")
        plt.plot(ep, mets["SWA"]["val"], "--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title("SPR_dataset – Training vs Validation SWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_dataset_SWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curves: {e}")
        plt.close()

    # 4) CmpWA curves ---------------------------------------------------------
    try:
        plt.figure()
        plt.plot(ep, mets["CmpWA"]["train"], label="Train")
        plt.plot(ep, mets["CmpWA"]["val"], "--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("CmpWA")
        plt.title("SPR_dataset – Training vs Validation CmpWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_dataset_CmpWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CmpWA curves: {e}")
        plt.close()

    # 5) Test-set summary bar chart ------------------------------------------
    try:
        plt.figure()
        names = ["CWA", "SWA", "CmpWA"]
        values = [test["CWA"], test["SWA"], test["CmpWA"]]
        bars = plt.bar(names, values, color=["skyblue", "salmon", "limegreen"])
        for bar, v in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
            )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_dataset – Test Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_dataset_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # -------- print test metrics ----------
    print(
        f"Test Loss={test['loss']:.4f} | CWA={test['CWA']:.4f} | "
        f"SWA={test['SWA']:.4f} | CmpWA={test['CmpWA']:.4f}"
    )
else:
    print("No experiment data found; nothing plotted.")
