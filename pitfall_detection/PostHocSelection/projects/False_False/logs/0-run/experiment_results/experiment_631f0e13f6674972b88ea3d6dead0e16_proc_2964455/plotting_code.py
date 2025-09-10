import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    exp = experiment_data["dropout_rate"]["SPR_BENCH"]
    rates = np.array(exp["rates"])
    train_f1 = np.array(exp["metrics"]["train_macroF1"])  # shape (R, E)
    val_f1 = np.array(exp["metrics"]["val_macroF1"])
    train_ls = np.array(exp["losses"]["train"])
    val_ls = np.array(exp["losses"]["val"])
    swa = np.array(exp["swa"])
    cwa = np.array(exp["cwa"])
    epochs = np.arange(1, train_f1.shape[1] + 1)

    # 1) Macro-F1 curves
    try:
        plt.figure()
        for i, r in enumerate(rates):
            plt.plot(epochs, train_f1[i], label=f"train d={r}")
            plt.plot(epochs, val_f1[i], "--", label=f"val d={r}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 vs Epochs\nSolid: Train, Dashed: Val")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 plot: {e}")
        plt.close()

    # 2) Loss curves
    try:
        plt.figure()
        for i, r in enumerate(rates):
            plt.plot(epochs, train_ls[i], label=f"train d={r}")
            plt.plot(epochs, val_ls[i], "--", label=f"val d={r}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss vs Epochs\nSolid: Train, Dashed: Val")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # 3) Final Val Macro-F1 vs Dropout
    try:
        plt.figure()
        final_val_f1 = val_f1[:, -1]
        plt.bar(rates, final_val_f1, width=0.05)
        plt.xlabel("Dropout Rate")
        plt.ylabel("Final Val Macro-F1")
        plt.title("SPR_BENCH Final Validation Macro-F1 vs Dropout")
        fname = os.path.join(working_dir, "SPR_BENCH_valF1_vs_dropout.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Val-F1 bar plot: {e}")
        plt.close()

    # 4) SWA vs Dropout
    try:
        plt.figure()
        plt.plot(rates, swa, marker="o")
        plt.xlabel("Dropout Rate")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH Shape-Weighted Accuracy vs Dropout")
        fname = os.path.join(working_dir, "SPR_BENCH_SWA_vs_dropout.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 5) CWA vs Dropout
    try:
        plt.figure()
        plt.plot(rates, cwa, marker="o", color="green")
        plt.xlabel("Dropout Rate")
        plt.ylabel("CWA")
        plt.title("SPR_BENCH Color-Weighted Accuracy vs Dropout")
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_vs_dropout.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # -------- Evaluation summary --------
    best_idx = np.argmax(final_val_f1)
    print(
        f"Best dropout rate: {rates[best_idx]:.2f} "
        f"with final Val Macro-F1={final_val_f1[best_idx]:.4f}"
    )
