import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------- load data ---------------------------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = exp["last_token_repr"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:

    # helper to unpack (epoch,value) pairs
    def unpack(pair_list):
        if not pair_list:
            return [], []
        ep, val = zip(*pair_list)
        return list(ep), list(val)

    # --------------- 1. contrastive pre-training loss ---------------
    try:
        ep, loss = unpack(spr["contrastive_pretrain"]["losses"])
        plt.figure()
        plt.plot(ep, loss, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        plt.title("SPR Contrastive Pre-training Loss")
        plt.savefig(os.path.join(working_dir, "SPR_contrastive_loss.png"))
        plt.close()
        print("Contrastive loss curve saved.")
    except Exception as e:
        print(f"Error creating contrastive loss plot: {e}")
        plt.close()

    # --------------- 2. fine-tune train / val loss ------------------
    try:
        ep_tr, tr = unpack(spr["fine_tune"]["losses"]["train"])
        ep_val, val = unpack(spr["fine_tune"]["losses"]["val"])
        plt.figure()
        plt.plot(ep_tr, tr, label="Train")
        plt.plot(ep_val, val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Fine-tune Losses")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_finetune_loss.png"))
        plt.close()
        print("Train/val loss curve saved.")
    except Exception as e:
        print(f"Error creating fine-tune loss plot: {e}")
        plt.close()

    # --------------- 3. metrics curves ------------------------------
    try:
        ep_swa, swa = unpack(spr["fine_tune"]["metrics"]["SWA"])
        _, cwa = unpack(spr["fine_tune"]["metrics"]["CWA"])
        _, comp = unpack(spr["fine_tune"]["metrics"]["CompWA"])
        plt.figure()
        plt.plot(ep_swa, swa, label="SWA")
        plt.plot(ep_swa, cwa, label="CWA")
        plt.plot(ep_swa, comp, label="CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR Fine-tune Metrics")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_metrics.png"))
        plt.close()
        print("Metrics curves saved.")
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # --------------- 4. accuracy per epoch --------------------------
    try:
        preds_pair = spr["fine_tune"]["predictions"]
        gts_pair = spr["fine_tune"]["ground_truth"]
        epochs = [ep for ep, _ in preds_pair]
        acc = []
        for (_, p), (_, g) in zip(preds_pair, gts_pair):
            p = np.array(p)
            g = np.array(g)
            acc.append((p == g).mean())
        plt.figure()
        plt.plot(epochs, acc, marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR Fine-tune Accuracy")
        plt.savefig(os.path.join(working_dir, "SPR_accuracy.png"))
        plt.close()
        print("Accuracy curve saved.")
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # --------------- 5. confusion matrix (last epoch) ---------------
    try:
        last_ep, last_preds = preds_pair[-1]
        _, last_gts = gts_pair[-1]
        classes = sorted(set(last_preds + last_gts))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(last_gts, last_preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"SPR Confusion Matrix (Epoch {last_ep})")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = f"SPR_confusion_matrix_epoch{last_ep}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print("Confusion matrix saved.")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
