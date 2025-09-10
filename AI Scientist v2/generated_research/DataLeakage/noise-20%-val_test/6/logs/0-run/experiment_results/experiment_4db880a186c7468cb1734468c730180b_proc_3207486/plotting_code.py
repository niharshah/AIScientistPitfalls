import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- Load experiment results ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    wd_dict = experiment_data.get("weight_decay", {})
    wds = sorted([float(k) for k in wd_dict.keys()])
    # Gather per-epoch traces
    epochs = len(next(iter(wd_dict.values()))["losses"]["train"])
    train_loss = {wd: wd_dict[str(wd)]["losses"]["train"] for wd in wds}
    val_loss = {wd: wd_dict[str(wd)]["losses"]["val"] for wd in wds}
    train_acc = {wd: wd_dict[str(wd)]["metrics"]["train_acc"] for wd in wds}
    val_acc = {wd: wd_dict[str(wd)]["metrics"]["val_acc"] for wd in wds}
    rule_fid = {wd: wd_dict[str(wd)]["metrics"]["rule_fidelity"] for wd in wds}
    test_accs = {wd: wd_dict[str(wd)]["test_acc"] for wd in wds}

    # --------- 1) Loss curves -------------
    try:
        plt.figure()
        for wd in wds:
            plt.plot(range(1, epochs + 1), train_loss[wd], label=f"train wd={wd}")
            plt.plot(
                range(1, epochs + 1), val_loss[wd], linestyle="--", label=f"val wd={wd}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # --------- 2) Accuracy curves ---------
    try:
        plt.figure()
        for wd in wds:
            plt.plot(range(1, epochs + 1), train_acc[wd], label=f"train wd={wd}")
            plt.plot(
                range(1, epochs + 1), val_acc[wd], linestyle="--", label=f"val wd={wd}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Training vs Validation Accuracy")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

    # --------- 3) Rule fidelity -----------
    try:
        plt.figure()
        for wd in wds:
            plt.plot(range(1, epochs + 1), rule_fid[wd], label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH: Rule-Fidelity Across Epochs")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating rule-fidelity plot: {e}")
        plt.close()

    # --------- 4) Final test accuracy -----
    try:
        plt.figure()
        plt.bar([str(wd) for wd in wds], [test_accs[wd] for wd in wds])
        plt.xlabel("Weight Decay")
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH: Final Test Accuracy vs. Weight Decay")
        fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar: {e}")
        plt.close()

    # --------- Print summary --------------
    print("\nFinal Test Accuracies:")
    for wd in wds:
        print(f"  weight_decay={wd:<6}: {test_accs[wd]:.3f}")
