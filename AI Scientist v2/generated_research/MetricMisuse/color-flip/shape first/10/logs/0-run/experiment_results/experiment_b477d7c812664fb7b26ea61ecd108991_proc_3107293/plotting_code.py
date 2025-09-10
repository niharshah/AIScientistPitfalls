import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_tags = list(experiment_data.keys())
dataset_name = "SPR"  # only dataset in dict

# ------------------------------------------------------------
# 1. Contrastive loss curves ---------------------------------
try:
    plt.figure()
    for tag in exp_tags:
        losses = experiment_data[tag][dataset_name]["losses"].get("contrastive", [])
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, label=tag)
    plt.xlabel("Pre-training epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name}: Contrastive Pre-training Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_contrastive_loss.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating contrastive loss plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 2. Supervised train / val loss curves ----------------------
try:
    plt.figure()
    for tag in exp_tags:
        train_l = experiment_data[tag][dataset_name]["losses"].get("train_sup", [])
        val_l = experiment_data[tag][dataset_name]["losses"].get("val_sup", [])
        epochs = experiment_data[tag][dataset_name].get(
            "epochs", list(range(1, len(train_l) + 1))
        )
        if train_l:
            plt.plot(epochs, train_l, linestyle="--", label=f"{tag} train")
        if val_l:
            plt.plot(epochs, val_l, linestyle="-", label=f"{tag} val")
    plt.xlabel("Fine-tuning epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name}: Supervised Train/Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_sup_loss.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating supervised loss plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 3. Augmentation Consistency Score vs epoch -----------------
try:
    plt.figure()
    for tag in exp_tags:
        acs = experiment_data[tag][dataset_name]["metrics"].get("val_ACS", [])
        if acs:
            plt.plot(range(1, len(acs) + 1), acs, marker="o", label=tag)
    plt.xlabel("Fine-tuning epoch")
    plt.ylabel("ACS")
    plt.title(f"{dataset_name}: Validation Augmentation-Consistency Score")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_ACS_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating ACS curve plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 4. Final ACS bar comparison --------------------------------
try:
    plt.figure()
    tags, finals = [], []
    for tag in exp_tags:
        acs = experiment_data[tag][dataset_name]["metrics"].get("val_ACS", [])
        if acs:
            tags.append(tag)
            finals.append(acs[-1])
    if finals:
        plt.bar(tags, finals, color=["steelblue", "orange"])
        plt.ylabel("ACS")
        plt.title(f"{dataset_name}: Final Augmentation-Consistency Score")
        for i, v in enumerate(finals):
            plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        fname = os.path.join(working_dir, f"{dataset_name}_ACS_final.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating final ACS bar plot: {e}")
    plt.close()
