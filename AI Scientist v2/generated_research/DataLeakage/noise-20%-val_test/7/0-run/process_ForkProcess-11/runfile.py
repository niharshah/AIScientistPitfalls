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

# containers for cross-dataset comparison
ds_names, ds_test_accs, ds_test_rfs = [], [], []

for ds_name, ds_dict in experiment_data.items():
    metrics = ds_dict.get("metrics", {})
    losses = ds_dict.get("losses", {})
    preds = np.asarray(ds_dict.get("predictions", []))
    gts = np.asarray(ds_dict.get("ground_truth", []))
    rule_p = np.asarray(ds_dict.get("rule_preds", []))
    test_acc = ds_dict.get("test_acc", None)
    test_rfs = ds_dict.get("test_rfs", None)
    if test_acc is not None:
        ds_names.append(ds_name)
        ds_test_accs.append(test_acc)
    if test_rfs is not None:
        ds_test_rfs.append(test_rfs)
    # ----------- detect whether metrics are nested by hidden dim ------------
    flat_layout = "train_acc" in metrics
    if flat_layout:
        hdims = [None]  # single run, treat as None
    else:
        hdims = sorted([hd for hd in metrics if isinstance(hd, (int, str))])[:5]  # ≤5
    # -------------------- ACCURACY CURVES -----------------------------------
    try:
        plt.figure(figsize=(6, 4))
        if flat_layout:
            epochs = np.arange(1, len(metrics["train_acc"]) + 1)
            plt.plot(epochs, metrics["train_acc"], label="train")
            plt.plot(epochs, metrics["val_acc"], "--", label="val")
        else:
            for hd in hdims:
                ep = np.arange(1, len(metrics[hd]["train_acc"]) + 1)
                plt.plot(ep, metrics[hd]["train_acc"], label=f"{hd}-train")
                plt.plot(ep, metrics[hd]["val_acc"], "--", label=f"{hd}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name}: Training vs Validation Accuracy")
        plt.legend(fontsize=7, ncol=2)
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_acc_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves for {ds_name}: {e}")
        plt.close()
    # -------------------- LOSS CURVES ---------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        if flat_layout:
            epochs = np.arange(1, len(metrics["val_loss"]) + 1)
            plt.plot(epochs, losses["train"], label="train")
            plt.plot(epochs, metrics["val_loss"], "--", label="val")
        else:
            for hd in hdims:
                ep = np.arange(1, len(metrics[hd]["val_loss"]) + 1)
                plt.plot(ep, losses[hd]["train"], label=f"{hd}-train")
                plt.plot(ep, metrics[hd]["val_loss"], "--", label=f"{hd}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name}: Training vs Validation Loss")
        plt.legend(fontsize=7, ncol=2)
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_loss_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds_name}: {e}")
        plt.close()
    # --------------- FINAL VAL ACCURACY PER HIDDEN DIM ----------------------
    if not flat_layout:
        try:
            plt.figure(figsize=(5, 3))
            final_va = [metrics[hd]["val_acc"][-1] for hd in hdims]
            plt.bar([str(hd) for hd in hdims], final_va, color="skyblue")
            plt.xlabel("Hidden Dim")
            plt.ylabel("Final Val Acc")
            plt.title(f"{ds_name}: Final Val Accuracy per Hidden Size")
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_val_acc_bar.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            print(f"Error creating val‐acc bar for {ds_name}: {e}")
            plt.close()
    # --------------------- CONFUSION MATRIX ---------------------------------
    try:
        if preds.size and gts.size:
            classes = sorted(np.unique(np.concatenate([gts, preds])))
            cm = np.zeros((len(classes), len(classes)), int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{ds_name}: Confusion Matrix (Test)")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()
    # ------------------- ACCURACY VS FIDELITY -------------------------------
    try:
        if (test_acc is not None) and (test_rfs is not None):
            plt.figure(figsize=(4, 3))
            plt.bar(
                ["Test Acc", "Rule Fidelity"],
                [test_acc, test_rfs],
                color=["green", "orange"],
            )
            plt.ylim(0, 1)
            plt.title(f"{ds_name}: Test Accuracy vs Rule Fidelity")
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_acc_vs_fidelity.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating acc vs fidelity for {ds_name}: {e}")
        plt.close()
    # ------------------- PRINT METRICS --------------------------------------
    if (test_acc is not None) and (test_rfs is not None):
        print(f"{ds_name}: TestAcc={test_acc:.4f}  RuleFidelity={test_rfs:.4f}")

# ------------------- CROSS-DATASET COMPARISON ------------------------------
try:
    if len(ds_names) >= 2:
        plt.figure(figsize=(5, 3))
        plt.bar(ds_names, ds_test_accs, color="purple")
        plt.ylim(0, 1)
        plt.ylabel("Test Accuracy")
        plt.title("Datasets: Test Accuracy Comparison")
        plt.savefig(
            os.path.join(working_dir, "comparison_test_acc.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        if len(ds_test_rfs) == len(ds_names):
            plt.figure(figsize=(5, 3))
            plt.bar(ds_names, ds_test_rfs, color="teal")
            plt.ylim(0, 1)
            plt.ylabel("Rule Fidelity")
            plt.title("Datasets: Rule Fidelity Comparison")
            plt.savefig(
                os.path.join(working_dir, "comparison_rule_fidelity.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
except Exception as e:
    print(f"Error creating cross-dataset plots: {e}")
    plt.close()
