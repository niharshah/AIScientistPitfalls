import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def main():
    try:
        experiment_data = np.load(
            os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
        ).item()
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return

    for dset, dct in experiment_data.items():
        epochs = np.array(dct.get("epochs", []))
        train_loss = np.array(dct.get("losses", {}).get("train", []))
        val_loss = np.array(dct.get("losses", {}).get("val", []))

        # metrics list of dicts -> dict of lists
        val_metrics = dct.get("metrics", {}).get("val", [])
        acc = (
            np.array([m.get("acc") for m in val_metrics])
            if val_metrics
            else np.array([])
        )
        pcwa = (
            np.array([m.get("PCWA") for m in val_metrics])
            if val_metrics
            else np.array([])
        )
        cwa = (
            np.array([m.get("CWA") for m in val_metrics])
            if val_metrics
            else np.array([])
        )
        swa = (
            np.array([m.get("SWA") for m in val_metrics])
            if val_metrics
            else np.array([])
        )

        preds = np.array(dct.get("predictions", []))
        gts = np.array(dct.get("ground_truth", []))

        # 1) loss curves
        try:
            plt.figure()
            if train_loss.size:
                plt.plot(epochs, train_loss, label="Train")
            if val_loss.size:
                plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset}: Train vs Val Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset}: {e}")
            plt.close()

        # 2) accuracy curve
        try:
            if acc.size:
                plt.figure()
                plt.plot(epochs, acc, marker="o")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"{dset}: Validation Accuracy")
                fname = os.path.join(working_dir, f"{dset}_val_accuracy.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {dset}: {e}")
            plt.close()

        # 3) specialised metric curves
        try:
            if pcwa.size and cwa.size and swa.size:
                plt.figure()
                plt.plot(epochs, pcwa, label="PCWA")
                plt.plot(epochs, cwa, label="CWA")
                plt.plot(epochs, swa, label="SWA")
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title(f"{dset}: Validation PCWA/CWA/SWA")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset}_val_special_metrics.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating special metrics plot for {dset}: {e}")
            plt.close()

        # 4) confusion matrix
        try:
            if preds.size and gts.size:
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

                cm = confusion_matrix(gts, preds)
                disp = ConfusionMatrixDisplay(cm)
                disp.plot()
                plt.title(f"{dset}: Confusion Matrix")
                fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix plot for {dset}: {e}")
            plt.close()

        # print final epoch metrics
        if val_metrics:
            print(f"Final {dset} metrics:", val_metrics[-1])


if __name__ == "__main__":
    main()
