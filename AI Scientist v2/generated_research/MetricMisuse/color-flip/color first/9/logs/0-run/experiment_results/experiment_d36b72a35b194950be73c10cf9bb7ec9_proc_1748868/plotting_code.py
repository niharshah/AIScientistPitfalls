import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------------------- 1) LOSS CURVES PER RULE -------------------
for rule, data in list(experiment_data.get("single_dataset", {}).items())[
    :5
]:  # max 4 rules
    try:
        train_l = data["losses"]["train"]
        val_l = data["losses"]["val"]
        epochs = range(1, len(train_l) + 1)

        plt.figure()
        plt.plot(epochs, train_l, label="Train Loss")
        plt.plot(epochs, val_l, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{rule} Dataset – Training vs Validation Loss")
        plt.legend()
        fname = f"{rule}_dataset_loss_curve.png".replace(" ", "_")
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {rule}: {e}")
        plt.close()

# -------- 2) BAR CHART: SINGLE vs UNION TEST BWA PER RULE -------
try:
    rules = list(experiment_data.get("single_dataset", {}).keys())
    single_bwa = [
        experiment_data["single_dataset"][r]["metrics"]["test"][-1] for r in rules
    ]  # index -1 = BWA
    union_bwa = [experiment_data["union_all"][r]["metrics"]["test"][-1] for r in rules]

    x = np.arange(len(rules))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, single_bwa, width, label="Single")
    plt.bar(x + width / 2, union_bwa, width, label="Union")
    plt.xticks(x, rules, rotation=45, ha="right")
    plt.ylabel("Balanced Weighted Accuracy")
    plt.title("Test BWA – Single vs UNION Models")
    plt.legend()
    plt.tight_layout()
    fname = "BWA_single_vs_union.png"
    plt.savefig(os.path.join(working_dir, fname))
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating BWA comparison plot: {e}")
    plt.close()
