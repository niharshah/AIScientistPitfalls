import matplotlib.pyplot as plt
import numpy as np
import os

# --------- SETUP ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- LOAD DATA -------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["tfidf_ngram"]["SPR_BENCH"]
    cfgs = ed["configs"]
    epochs = len(ed["metrics"]["train_acc"][0])
    xs = np.arange(1, epochs + 1)

    # ----- PER-CONFIG CURVES -----
    for i, cfg in enumerate(cfgs):
        # Accuracy
        try:
            plt.figure()
            plt.plot(xs, ed["metrics"]["train_acc"][i], label="Train")
            plt.plot(xs, ed["metrics"]["val_acc"][i], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"SPR_BENCH Accuracy – {cfg}")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"SPR_BENCH_{cfg}_accuracy.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot ({cfg}): {e}")
            plt.close()

        # Loss
        try:
            plt.figure()
            plt.plot(xs, ed["losses"]["train"][i], label="Train")
            plt.plot(xs, ed["losses"]["val"][i], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"SPR_BENCH Loss – {cfg}")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"SPR_BENCH_{cfg}_loss.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot ({cfg}): {e}")
            plt.close()

        # Rule fidelity
        try:
            plt.figure()
            plt.plot(xs, ed["metrics"]["rule_fidelity"][i], label="Rule Fidelity")
            plt.xlabel("Epoch")
            plt.ylabel("Fidelity")
            plt.title(f"SPR_BENCH Rule Fidelity – {cfg}")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"SPR_BENCH_{cfg}_rule_fid.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating rule fidelity plot ({cfg}): {e}")
            plt.close()

    # ----- CONFUSION MATRIX FOR BEST CONFIG -----
    try:
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        n_cls = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f'SPR_BENCH Confusion Matrix – {ed["best_config"]}')
        for r in range(n_cls):
            for c in range(n_cls):
                plt.text(
                    c, r, cm[r, c], va="center", ha="center", color="black", fontsize=7
                )
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best.png"))
        plt.close()

        test_acc = (preds == gts).mean()
        print(f'Best config: {ed["best_config"]} | Test accuracy: {test_acc:.3f}')
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
