import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ---------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["no_label_smoothing"]["SPR_BENCH"]
    epochs = run["epochs"]
    tr_loss, va_loss = run["losses"]["train"], run["losses"]["val"]
    tr_metrics, va_metrics = run["metrics"]["train"], run["metrics"]["val"]
    tr_acc = [m["acc"] for m in tr_metrics]
    va_acc = [m["acc"] for m in va_metrics]
    tr_mcc = [m["MCC"] for m in tr_metrics]
    va_mcc = [m["MCC"] for m in va_metrics]
    tr_rma = [m["RMA"] for m in tr_metrics]
    va_rma = [m["RMA"] for m in va_metrics]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    epochs, tr_loss, va_loss, tr_acc, va_acc, tr_mcc, va_mcc, tr_rma, va_rma = ([],) * 9


# -------------- helper for plotting -------------- #
def plot_two_lines(x, y1, y2, title, ylabel, fname):
    try:
        plt.figure()
        plt.plot(x, y1, marker="o", label="Train")
        plt.plot(x, y2, marker="s", label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {fname}: {e}")
        plt.close()


# -------------- create plots -------------- #
if epochs:
    plot_two_lines(
        epochs,
        tr_loss,
        va_loss,
        "SPR_BENCH – Train vs Val Loss",
        "BCE Loss",
        "SPR_BENCH_loss_curve_no_label_smoothing.png",
    )

    plot_two_lines(
        epochs,
        tr_acc,
        va_acc,
        "SPR_BENCH – Train vs Val Accuracy",
        "Accuracy",
        "SPR_BENCH_accuracy_curve_no_label_smoothing.png",
    )

    plot_two_lines(
        epochs,
        tr_mcc,
        va_mcc,
        "SPR_BENCH – Train vs Val MCC",
        "Matthews CorrCoef",
        "SPR_BENCH_MCC_curve_no_label_smoothing.png",
    )

    plot_two_lines(
        epochs,
        tr_rma,
        va_rma,
        "SPR_BENCH – Train vs Val Rule-Macro Acc",
        "Rule-Macro Accuracy",
        "SPR_BENCH_RMA_curve_no_label_smoothing.png",
    )
else:
    print("No epoch data to plot.")
