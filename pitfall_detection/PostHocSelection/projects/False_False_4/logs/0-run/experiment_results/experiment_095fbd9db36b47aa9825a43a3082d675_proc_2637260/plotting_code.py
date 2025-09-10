import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

lrs = sorted(experiment_data.get("learning_rate", {}).keys(), key=float)

# ---------- Plot 1: Loss curves ----------
try:
    plt.figure()
    for lr in lrs:
        d = experiment_data["learning_rate"][lr]
        plt.plot(d["losses"]["train"], label=f"train lr={lr}")
        plt.plot(d["losses"]["val"], label=f"val   lr={lr}", linestyle="--")
    plt.title("SPR_Bench Loss Curves\nTrain vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend(fontsize=7)
    plt.savefig(os.path.join(working_dir, "SPR_Bench_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- Plot 2: Accuracy curves ----------
try:
    plt.figure()
    for lr in lrs:
        d = experiment_data["learning_rate"][lr]
        plt.plot(d["metrics"]["train"], label=f"train lr={lr}")
        plt.plot(d["metrics"]["val"], label=f"val   lr={lr}", linestyle="--")
    plt.title("SPR_Bench Accuracy Curves\nTrain vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=7)
    plt.savefig(os.path.join(working_dir, "SPR_Bench_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# ---------- Plot 3: RGS curves ----------
try:
    plt.figure()
    for lr in lrs:
        d = experiment_data["learning_rate"][lr]
        plt.plot(d["metrics"]["val_rgs"], label=f"val RGS lr={lr}")
    plt.title("SPR_Bench Validation RGS Curves")
    plt.xlabel("Epoch")
    plt.ylabel("RGS")
    plt.legend(fontsize=7)
    plt.savefig(os.path.join(working_dir, "SPR_Bench_RGS_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RGS curves: {e}")
    plt.close()

# ---------- Plot 4: Final test accuracy ----------
try:
    accs = [experiment_data["learning_rate"][lr]["test_metrics"]["acc"] for lr in lrs]
    x = np.arange(len(lrs))
    plt.figure()
    plt.bar(x, accs, color="skyblue")
    plt.xticks(x, lrs)
    plt.title("SPR_Bench Final Test Accuracy per Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "SPR_Bench_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()

# ---------- Plot 5: Final test RGS, SWA, CWA ----------
try:
    width = 0.25
    x = np.arange(len(lrs))
    rgs = [experiment_data["learning_rate"][lr]["test_metrics"]["rgs"] for lr in lrs]
    swa = [experiment_data["learning_rate"][lr]["test_metrics"]["swa"] for lr in lrs]
    cwa = [experiment_data["learning_rate"][lr]["test_metrics"]["cwa"] for lr in lrs]
    plt.figure()
    plt.bar(x - width, rgs, width=width, label="RGS")
    plt.bar(x, swa, width=width, label="SWA")
    plt.bar(x + width, cwa, width=width, label="CWA")
    plt.xticks(x, lrs)
    plt.title("SPR_Bench Test Metrics\nRGS vs SWA vs CWA")
    plt.xlabel("Learning Rate")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_Bench_test_RGS_SWA_CWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating grouped test metrics plot: {e}")
    plt.close()
