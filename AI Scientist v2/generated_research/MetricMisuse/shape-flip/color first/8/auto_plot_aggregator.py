import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
os.makedirs("figures", exist_ok=True)

TITLE_FONT = 16
LABEL_FONT = 14
LEGEND_FONT = 12
DPI = 300

def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_loss_and_accuracy(exp_name, epochs_loss, loss_train, loss_val, 
                             epochs_acc, acc_train, acc_val, save_path):
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
        # Loss subplot
        axs[0].plot(epochs_loss, loss_train, label="Train", marker="o")
        axs[0].plot(epochs_loss, loss_val, label="Validation", marker="o")
        axs[0].set_xlabel("Epoch", fontsize=LABEL_FONT)
        axs[0].set_ylabel("Cross Entropy Loss", fontsize=LABEL_FONT)
        axs[0].set_title(f"{exp_name} Loss Curve", fontsize=TITLE_FONT)
        axs[0].legend(fontsize=LEGEND_FONT)
        remove_top_right_spines(axs[0])
        # Accuracy subplot
        axs[1].plot(epochs_acc, acc_train, label="Train", marker="o")
        axs[1].plot(epochs_acc, acc_val, label="Validation", marker="o")
        axs[1].set_xlabel("Epoch", fontsize=LABEL_FONT)
        axs[1].set_ylabel("Accuracy", fontsize=LABEL_FONT)
        axs[1].set_title(f"{exp_name} Accuracy Curve", fontsize=TITLE_FONT)
        axs[1].legend(fontsize=LEGEND_FONT)
        remove_top_right_spines(axs[1])
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved aggregated loss and accuracy figure: {save_path}")
    except Exception as e:
        print(f"Error plotting aggregated loss and accuracy for {exp_name}: {e}")
        plt.close()

def plot_confusion_matrix(exp_name, preds, gts, save_path):
    try:
        if not preds or not gts:
            print(f"Skipping confusion matrix for {exp_name} due to missing data.")
            return
        preds = np.array(preds)
        gts = np.array(gts)
        num_classes = int(max(np.max(preds), np.max(gts))) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for actual, pred in zip(gts, preds):
            cm[actual, pred] += 1
        fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI)
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted", fontsize=LABEL_FONT)
        ax.set_ylabel("Ground Truth", fontsize=LABEL_FONT)
        ax.set_title(f"{exp_name} Confusion Matrix", fontsize=TITLE_FONT)
        ticks = list(range(num_classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"Class {i}" for i in ticks], fontsize=LEGEND_FONT)
        ax.set_yticklabels([f"Class {i}" for i in ticks], fontsize=LEGEND_FONT)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=LEGEND_FONT, color="white" if cm[i, j] > cm.max()/2 else "black")
        remove_top_right_spines(ax)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved confusion matrix figure: {save_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix for {exp_name}: {e}")
        plt.close()

def safe_load(filepath):
    try:
        data = np.load(filepath, allow_pickle=True).item()
        print(f"Loaded data from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return {}

def extract_record(data):
    """
    Recursively search for a record which has keys 'losses' and 'metrics'
    """
    if isinstance(data, dict):
        if "losses" in data and "metrics" in data:
            return data
        for key in data:
            rec = extract_record(data[key])
            if rec is not None:
                return rec
    return None

# Map of experiment display names to their npy file paths (exact paths from summaries)
experiment_files = {
    "Baseline Research": "experiment_results/experiment_2e530e6554fd432c9557ba4fa368902d_proc_1509391/experiment_data.npy",
    "Shape Only": "experiment_results/experiment_02dbf617f2144ebb9ee988aa89ddcc09_proc_1518069/experiment_data.npy",
    "Sequence Order Shuffled": "experiment_results/experiment_b1ab23abaa734c2d8219cbb014adaa57_proc_1518070/experiment_data.npy",
    "Fully Connected": "experiment_results/experiment_e6d55690903a4ab79898600856655bd5_proc_1518071/experiment_data.npy",
    "Depth 1 GCN": "experiment_results/experiment_a21f036dfd9e462a8f573b71928f0a7c_proc_1518068/experiment_data.npy",
    "Directed Edges": "experiment_results/experiment_3b47406c07244c498f4f00c456d15bd2_proc_1518070/experiment_data.npy"
}

# Create at most 12 unique figures (each experiment up to 2 figures)
for exp_display, filepath in experiment_files.items():
    data = safe_load(filepath)
    if not data:
        continue
    # Try direct key lookup for "SPR_BENCH"
    rec = data.get("SPR_BENCH", None)
    if rec is None:
        # Try alternate key for shape_only experiment
        if exp_display == "Shape Only" and "shape_only" in data:
            rec = data["shape_only"].get("SPR_BENCH", None)
    if rec is None:
        rec = extract_record(data)
    if rec is None:
        print(f"Record with losses and metrics not found for {exp_display}. Skipping.")
        continue

    loss_train = rec.get("losses", {}).get("train", [])
    loss_val = rec.get("losses", {}).get("val", [])
    acc_train = rec.get("metrics", {}).get("train", [])
    acc_val = rec.get("metrics", {}).get("val", [])
    predictions = rec.get("predictions", [])
    ground_truth = rec.get("ground_truth", [])

    if loss_train and loss_val:
        epochs_loss = list(range(1, len(loss_train) + 1))
    else:
        epochs_loss = []
    if acc_train and acc_val:
        epochs_acc = list(range(1, len(acc_train) + 1))
    else:
        epochs_acc = []

    if epochs_loss and epochs_acc:
        plot_loss_and_accuracy(exp_display, epochs_loss, loss_train, loss_val,
                               epochs_acc, acc_train, acc_val,
                               os.path.join("figures", f"{exp_display.replace(' ', '_')}_loss_accuracy.png"))
    else:
        print(f"Missing loss or accuracy data for {exp_display}.")

    if predictions and ground_truth:
        plot_confusion_matrix(exp_display, predictions, ground_truth,
                              os.path.join("figures", f"{exp_display.replace(' ', '_')}_confusion_matrix.png"))
    else:
        print(f"Missing prediction data for {exp_display}.")