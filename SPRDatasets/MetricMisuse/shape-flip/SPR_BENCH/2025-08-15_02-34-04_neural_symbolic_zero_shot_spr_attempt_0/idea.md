## Name

neural_symbolic_zero_shot_spr

## Title

Zero-Shot Synthetic PolyRule Reasoning with Neural Symbolic Integration

## Short Hypothesis

Integrating neural networks with symbolic reasoning frameworks enables zero-shot learning in Synthetic PolyRule Reasoning (SPR), allowing models to generalize to unseen rules without additional training.

## Related Work

1. 'Large Language Models are Zero-Shot Reasoners' by Kojima et al. highlights the zero-shot reasoning capabilities of LLMs using Chain-of-Thought (CoT) prompting. 2. 'Knowledge-driven Data Construction for Zero-shot Evaluation in Commonsense Question Answering' by Ma et al. discusses a neuro-symbolic framework for zero-shot QA. 3. 'Relational reasoning and generalization using non-symbolic neural networks' by Geiger et al. shows that neural networks can learn abstract relational reasoning.

## Abstract

This proposal aims to develop a novel algorithm that integrates neural networks with symbolic reasoning frameworks to achieve zero-shot learning in Synthetic PolyRule Reasoning (SPR). The key innovation is the use of a neural-symbolic model that can infer and apply new rules without additional training, allowing it to generalize to unseen tasks. We will evaluate the approach using the SPR_BENCH benchmark, focusing on its ability to surpass state-of-the-art performance in Shape-Weighted Accuracy (SWA) and Color-Weighted Accuracy (CWA) metrics. If successful, this approach could revolutionize automated reasoning systems by enabling them to adapt to new, complex rules without the need for retraining.

## Experiments

- 1. Model Design: Develop a neural-symbolic model that combines a neural network for feature extraction with a symbolic reasoning component for rule inference.
- 2. Training: Train the model on a subset of SPR_BENCH with known rules and evaluate its performance on this training set.
- 3. Zero-Shot Evaluation: Test the model on sequences governed by entirely new rules not seen during training. Choose only **one** evaluation metric either Shape-Weighted Accuracy (SWA) or Color-Weighted Accuracy (CWA) for performance comparison.
- 4. Ablation Study: Conduct an ablation study to assess the contribution of the neural and symbolic components individually.

## Risk Factors And Limitations

1. Model Complexity: Integrating neural networks with symbolic reasoning frameworks could lead to increased model complexity, making training and inference more computationally intensive. 2. Rule Generalization: The model's ability to generalize to unseen rules is uncertain and may require careful design of the symbolic reasoning component. 3. Benchmark Limitations: The SPR_BENCH benchmark's predefined rules may not fully capture the diversity of real-world symbolic reasoning tasks, potentially limiting the generalizability of the results.

## Code To Potentially Use

Use the following code as context for your experiments:

```python
"""
SPR.py
────────────────────────────────────────────────────────
Utility to load the SPR_BENCH benchmark datasets
Using HuggingFace’s `datasets` library.

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


def count_shape_variety(sequence: str) -> int:
    """Count the number of unique shape types in the sequence"""
    return len(set(token[0] for token in sequence.strip().split() if token))

def count_color_variety(sequence: str) -> int:
    """Count the number of unique color types in the sequence"""
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))

def shape_weighted_accuracy(sequences, y_true, y_pred):
    """Shape-Weighted Accuracy (SWA)"""
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

def color_weighted_accuracy(sequences, y_true, y_pred):
    """Color-Weighted Accuracy (CWA)"""
    weights = [count_color_variety(seq) for seq in sequences]
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

