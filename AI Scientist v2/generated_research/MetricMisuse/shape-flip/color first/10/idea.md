## Name

gnn_for_spr

## Title

Leveraging Graph Neural Networks for Enhanced Synthetic PolyRule Reasoning

## Short Hypothesis

Graph Neural Networks (GNNs) can effectively capture the inherent structure and relationships within sequences of symbolic data in the Synthetic PolyRule Reasoning (SPR) task, leading to superior performance on both Color-Weighted Accuracy (CWA) and Shape-Weighted Accuracy (SWA) metrics compared to current State-of-the-Art (SOTA) models.

## Related Work

Current approaches for symbolic pattern recognition often rely on sequence models like RNNs, LSTMs, and Transformers. These models primarily focus on the sequential nature of the data but may not fully exploit the relational and structural information present in the sequences. While some works have explored GNNs for related tasks, they have not been specifically applied to the SPR task involving complex poly-factor rules. This proposal aims to fill this gap by leveraging the unique strengths of GNNs in capturing structural dependencies and relationships within the sequences.

## Abstract

This research proposes the use of Graph Neural Networks (GNNs) for the Synthetic PolyRule Reasoning (SPR) task. The SPR task involves classifying sequences of symbolic data according to hidden poly-factor rules. Current SOTA models primarily utilize sequence-based architectures that may not fully capture the relational and structural information inherent in the sequences. We hypothesize that GNNs, designed to model relational data, can better capture these dependencies, leading to improved performance on both Color-Weighted Accuracy (CWA) and Shape-Weighted Accuracy (SWA) metrics. We will design a GNN-based model that represents each sequence as a graph, with nodes corresponding to tokens and edges representing relationships based on position, color, and shape. The model will be evaluated on the SPR_BENCH benchmark, and we aim to surpass the current SOTA performance on the chosen metric.

## Experiments

- {'Name': 'Model Design and Training', 'Description': 'Design a GNN-based model where each token in the sequence is represented as a node, and edges capture relationships based on color, shape, position, and order. Train the model on the training split of the SPR_BENCH benchmark. Tune hyperparameters using the development split.'}
- {'Name': 'Evaluation', 'Description': "Evaluate the model's performance on the test split using Color-Weighted Accuracy (CWA) and Shape-Weighted Accuracy (SWA). Compare the performance against the current SOTA benchmarks (CWA: 65.0%, SWA: 70.0%)."}
- {'Name': 'Ablation Studies', 'Description': "Investigate the impact of different types of edges (e.g., color-based, shape-based, position-based) on the model's performance. Compare the performance of the GNN-based model to traditional sequence-based models (e.g., LSTMs, Transformers)."}
- {'Name': 'Visualization and Analysis', 'Description': 'Visualize the learned graph embeddings to understand how the model captures the relational structure of the sequences. Analyze the types of rules that the model learns to recognize, and identify any patterns or commonalities among misclassified instances.'}

## Risk Factors And Limitations

- Data Representation: Representing sequences as graphs introduces complexity in data preprocessing. Ensuring that the graph representation accurately captures all relevant relationships is crucial.
- Scalability: GNNs can be computationally intensive, especially for longer sequences with many nodes and edges. Efficient implementation and optimization will be necessary to handle large datasets.
- Generalization: While GNNs are powerful, there is a risk that the model may overfit to specific patterns in the training data. Careful regularization and validation will be required to ensure robust generalization.

## Code To Potentially Use

Use the following code as context for your experiments:

```python
"""
SPR.py
────────────────────────────────────────────────────────
Utility to load the SPR_BENCH benchmark datasets
Using HuggingFace’s `datasets` library.
Definition of two evaluation metrics:
1. Color-Weighted Accuracy (CWA)
2. Shape-Weighted Accuracy (SWA)
Directory layout expected
SPR_BENCH/
 ├─ train.csv   (20000 rows)
 ├─ dev.csv     (5000 rows)
 └─ test.csv    (10000 rows)

Each CSV has header:  id,sequence,label
────────────────────────────────────────────────────────
$ pip install datasets   # once
"""
import pathlib
from typing import Dict

from datasets import load_dataset, DatasetDict                                         # <- no pandas import


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    """
    Return a DatasetDict {'train':…, 'dev':…, 'test':…} for one SPR ID folder.
    """
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",           # treat csv as a single split
            cache_dir=".cache_dsets" # optional; keeps HF cache tidy
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"]   = _load("dev.csv")
    dset["test"]  = _load("test.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    """Count the number of unique color types in the sequence"""
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    """Count the number of unique shape types in the sequence"""
    return len(set(token[0] for token in sequence.strip().split() if token))

def color_weighted_accuracy(sequences, y_true, y_pred):
    """Color-Weighted Accuracy (CWA)"""
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

def shape_weighted_accuracy(sequences, y_true, y_pred):
    """Shape-Weighted Accuracy (SWA)"""
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def main():

    ## Absolute path of the datasets
    DATA_PATH = pathlib.Path('/home/zxl240011/AI-Scientist-v2/SPR_BENCH/')
    spr_bench = load_spr_bench(DATA_PATH)

    print("Benchmarks split:", spr_bench.keys())

    # Demo: show first example from SPR_BENCH‑train
    ex = spr_bench["train"][0]
    print("\nExample row:")
    print(ex)          


if __name__ == "__main__":
    main()

```

