import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data["dropout_probability"]["SPR_BENCH"]
    dropouts = sorted(data.keys(), key=float)

    # pre-collect stats
    epochs = None
    train_loss, val_loss = {}, {}
    train_f1, val_f1 = {}, {}
    final_val_f1 = {}

    for dr in dropouts:
        rec = data[dr]
        tl = [d["loss"] for d in rec["losses"]["train"]]
        vl = [d["loss"] for d in rec["losses"]["val"]]
        tf = [d["macro_f1"] for d in rec["metrics"]["train"]]
        vf = [d["macro_f1"] for d in rec["metrics"]["val"]]

        train_loss[dr], val_loss[dr] = tl, vl
        train_f1[dr], val_f1[dr] = tf, vf
        final_val_f1[dr] = vf[-1]
        if epochs is None:
            epochs = [d["epoch"] for d in rec["metrics"]["train"]]

    # helper to plot multiple curves
    def multi_curve_plot(curves, ylabel, title, filename):
        try:
            plt.figure()
            for dr in dropouts:
                plt.plot(epochs, curves[dr], marker="o", label=f"dropout={dr}")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"SPR_BENCH {title}\nLines: different dropout probabilities")
            plt.legend()
            plt.grid(True, alpha=0.3)
            save_path = os.path.join(working_dir, filename)
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating {filename}: {e}")
            plt.close()

    # 1-4: curve plots
    multi_curve_plot(
        train_loss, "Loss", "Training Loss Curves", "spr_bench_train_loss.png"
    )
    multi_curve_plot(
        val_loss, "Loss", "Validation Loss Curves", "spr_bench_val_loss.png"
    )
    multi_curve_plot(
        train_f1, "Macro F1", "Training Macro-F1 Curves", "spr_bench_train_f1.png"
    )
    multi_curve_plot(
        val_f1, "Macro F1", "Validation Macro-F1 Curves", "spr_bench_val_f1.png"
    )

    # 5: bar chart of final val F1
    try:
        plt.figure()
        bars = [final_val_f1[dr] for dr in dropouts]
        plt.bar(dropouts, bars, color="skyblue")
        plt.ylim(0, 1)
        plt.xlabel("Dropout Probability")
        plt.ylabel("Final Validation Macro F1")
        plt.title("SPR_BENCH Final Validation Macro-F1\nOne bar per dropout setting")
        for i, v in enumerate(bars):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        save_path = os.path.join(working_dir, "spr_bench_final_val_f1_bar.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating final F1 bar plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # print summary metrics
    print("Final Validation Macro-F1 by Dropout:")
    for dr in dropouts:
        print(f"  dropout={dr}: {final_val_f1[dr]:.4f}")
