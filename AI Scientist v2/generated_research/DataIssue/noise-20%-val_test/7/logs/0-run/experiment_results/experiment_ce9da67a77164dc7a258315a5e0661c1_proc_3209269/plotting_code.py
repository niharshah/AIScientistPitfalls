import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- per-dataset visualisations ----------
for ds_name, ds in experiment_data.items():
    metrics, losses = ds.get("metrics", {}), ds.get("losses", {})
    train_acc, val_acc = metrics.get("train_acc", []), metrics.get("val_acc", [])
    train_loss, val_loss = losses.get("train", []), metrics.get("val_loss", [])
    preds, gts = np.asarray(ds.get("predictions", [])), np.asarray(
        ds.get("ground_truth", [])
    )
    rule_preds = np.asarray(ds.get("rule_preds", []))
    epochs = np.arange(1, len(train_acc) + 1)

    # 1) accuracy curves
    try:
        if len(train_acc) and len(val_acc):
            plt.figure()
            plt.plot(epochs, train_acc, label="train")
            plt.plot(epochs, val_acc, linestyle="--", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name}: Training vs Validation Accuracy")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_acc_curves.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves for {ds_name}: {e}")
        plt.close()

    # 2) loss curves
    try:
        if len(train_loss) and len(val_loss):
            plt.figure()
            plt.plot(epochs, train_loss, label="train")
            plt.plot(epochs, val_loss, linestyle="--", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name}: Training vs Validation Loss")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_loss_curves.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds_name}: {e}")
        plt.close()

    # 3) confusion matrix
    try:
        if preds.size and gts.size:
            classes = sorted(np.unique(np.concatenate([gts, preds])))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            plt.figure()
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

    # 4) accuracy vs fidelity
    try:
        test_acc = float((preds == gts).mean()) if preds.size else None
        fidelity = (
            float(metrics.get("rfs", [None])[-1]) if metrics.get("rfs", []) else None
        )
        if test_acc is not None and fidelity is not None:
            plt.figure()
            plt.bar(
                ["Test Acc", "Rule Fidelity"],
                [test_acc, fidelity],
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
        print(f"Error creating acc vs fidelity plot for {ds_name}: {e}")
        plt.close()

# ---------- aggregated comparison ----------
try:
    names, accs, fids = [], [], []
    for ds_name, ds in experiment_data.items():
        p, g = np.asarray(ds.get("predictions", [])), np.asarray(
            ds.get("ground_truth", [])
        )
        if p.size and g.size and ds.get("metrics", {}).get("rfs", []):
            names.append(ds_name)
            accs.append((p == g).mean())
            fids.append(ds["metrics"]["rfs"][-1])
    if names:
        x = np.arange(len(names))
        w = 0.35
        plt.figure()
        plt.bar(x - w / 2, accs, w, label="Test Acc")
        plt.bar(x + w / 2, fids, w, label="Rule Fidelity")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title("Datasets: Test Accuracy vs Rule Fidelity")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "datasets_acc_vs_fidelity.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated comparison plot: {e}")
    plt.close()

# ---------- print metrics ----------
for ds_name, ds in experiment_data.items():
    p, g = np.asarray(ds.get("predictions", [])), np.asarray(ds.get("ground_truth", []))
    if p.size and g.size and ds.get("metrics", {}).get("rfs", []):
        test_acc = (p == g).mean()
        fidelity = ds["metrics"]["rfs"][-1]
        print(
            f"{ds_name} - Test Accuracy: {test_acc:.4f} | Rule Fidelity: {fidelity:.4f}"
        )
