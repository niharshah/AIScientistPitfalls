import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- SETUP --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- DATA LOADING -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Exit early if data missing
if not experiment_data:
    quit()

# There is only one main entry under 'tree_depth_sensitivity'
root = experiment_data.get("tree_depth_sensitivity", {})
if not root:
    quit()
dataset_name = list(root.keys())[0]
data = root[dataset_name]

depth_labels = [str(d) for d in data["depths"]]
depth_xticks = ["∞" if d == "None" else str(d) for d in depth_labels]  # pretty print

# ------------------- PLOTS ---------------------
# 1. Accuracy plot
try:
    plt.figure()
    x = np.arange(len(depth_labels))
    plt.plot(x, data["metrics"]["train"], "o-", label="Train")
    plt.plot(x, data["metrics"]["val"], "s-", label="Validation")
    plt.plot(x, data["metrics"]["test"], "^-", label="Test")
    plt.xticks(x, depth_xticks)
    plt.ylabel("Accuracy")
    plt.xlabel("Tree Depth")
    plt.title(f"Accuracy vs. Depth ({dataset_name})\nTrain/Val/Test comparison")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_accuracy_vs_depth.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2. Loss plot
try:
    plt.figure()
    x = np.arange(len(depth_labels))
    plt.plot(x, data["losses"]["train"], "o-", label="Train")
    plt.plot(x, data["losses"]["val"], "s-", label="Validation")
    plt.xticks(x, depth_xticks)
    plt.ylabel("Log Loss")
    plt.xlabel("Tree Depth")
    plt.title(f"Log-Loss vs. Depth ({dataset_name})\nTrain/Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_loss_vs_depth.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3. Rule count plot
try:
    plt.figure()
    x = np.arange(len(depth_labels))
    plt.bar(x, data["rule_counts"])
    plt.xticks(x, depth_xticks)
    plt.ylabel("Number of Extracted Rules")
    plt.xlabel("Tree Depth")
    plt.title(f"Rule Count vs. Depth ({dataset_name})")
    fname = os.path.join(working_dir, f"{dataset_name}_rulecount_vs_depth.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating rule count plot: {e}")
    plt.close()
