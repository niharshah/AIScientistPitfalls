import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    beta2_dict = experiment_data.get("adam_beta2", {})
    # sort keys numerically for consistent order
    beta2_values = sorted(beta2_dict.keys(), key=lambda x: float(x))

    # --------- 1-5: loss curves, one per β2 -----------------
    for beta in beta2_values:
        try:
            data = beta2_dict[beta]["SPR_BENCH"]["losses"]
            tr_epochs, tr_losses = zip(*data["train"])
            val_epochs, val_losses = zip(*data["val"])

            plt.figure()
            plt.plot(tr_epochs, tr_losses, label="Train")
            plt.plot(val_epochs, val_losses, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"SPR_BENCH Loss Curve\nβ₂={beta}  |  Left: Train, Right: Val")
            plt.legend()
            fname = f"loss_curve_SPR_BENCH_beta2_{beta}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting loss for β2={beta}: {e}")
            plt.close()

    # --------- 6: summary bar chart of final val metrics ----
    try:
        metrics = ["CWA", "SWA", "EWA"]
        vals = {m: [] for m in metrics}
        for beta in beta2_values:
            metr_list = beta2_dict[beta]["SPR_BENCH"]["metrics"]["val"]
            _, last_dict = metr_list[-1]  # final epoch metrics
            for m in metrics:
                vals[m].append(last_dict[m])

        x = np.arange(len(beta2_values))
        width = 0.25
        plt.figure(figsize=(8, 4))
        for i, m in enumerate(metrics):
            plt.bar(x + i * width, vals[m], width, label=m)

        plt.xticks(x + width, beta2_values)
        plt.ylabel("Score")
        plt.title(
            "SPR_BENCH Final-Epoch Validation Metrics\nLeft to Right Bars: CWA, SWA, EWA"
        )
        plt.legend()
        fname = "val_metric_summary_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating summary metric plot: {e}")
        plt.close()
