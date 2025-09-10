## Name

context_aware_contrastive_learning

## Title

Context-aware Contrastive Learning for Enhanced Symbolic Pattern Recognition

## Short Hypothesis

Self-supervised contrastive learning, when enhanced with context-aware data augmentation and denoising techniques, will significantly improve the performance of models on the Synthetic PolyRule Reasoning (SPR) task by creating more robust and contextually aware feature representations of symbolic sequences.

## Related Work

1. Contrastive Learning: Contrastive learning has shown substantial improvements in learning effective feature representations across various domains (Chen et al., 2020; He et al., 2020). Studies like Guo et al. (2021) and Li et al. (2023) demonstrate how contrastive learning can be enhanced with advanced data augmentation and denoising techniques, which can be adapted for symbolic sequences.
2. Symbolic Reasoning: Traditional methods for symbolic reasoning often rely on extensive labeled data and struggle with generalization (Evans et al., 2018; Selsam et al., 2019). Incorporating contrastive learning can alleviate these issues by leveraging unlabeled data to learn more generalizable representations.
3. SPR_BENCH Benchmark: The SPR_BENCH dataset provides a standardized benchmark for evaluating symbolic pattern recognition models. Current SOTA methods rely on supervised learning, achieving 65.0% SWA and 70.0% CWA, leaving room for improvement with innovative approaches.

## Abstract

This research proposes leveraging context-aware self-supervised contrastive learning to enhance feature representations for the Synthetic PolyRule Reasoning (SPR) task. The SPR task involves classifying symbolic sequences governed by hidden logical rules, with significant potential for applications in automated reasoning systems. We hypothesize that context-aware contrastive learning, combined with advanced data augmentation and denoising techniques, can create more robust and generalizable embeddings of symbolic sequences. These embeddings can then be fine-tuned for the specific SPR task. We will design a context-aware contrastive learning framework tailored to the symbolic nature of the SPR_BENCH dataset and evaluate its performance using Shape-Weighted Accuracy (SWA) and Color-Weighted Accuracy (CWA) metrics. The proposed approach aims to surpass the current SOTA performance of 65.0% SWA and 70.0% CWA, demonstrating the effectiveness of contrastive learning in symbolic pattern recognition.

## Experiments

1. Context-aware Contrastive Learning Framework Design:
- Develop a context-aware contrastive learning framework tailored for symbolic sequences.
- Incorporate advanced data augmentation techniques (e.g., token shuffling, token masking) and denoising strategies to create positive and negative pairs based on sequence similarity and dissimilarity in terms of shape and color complexity.

2. Pre-training on Unlabeled Data:
- Pre-train the model on unlabeled sequences from the SPR_BENCH dataset using the context-aware contrastive learning framework.
- Evaluate the quality of learned embeddings using visualization techniques (e.g., t-SNE) and clustering metrics.

3. Fine-tuning for SPR Task:
- Fine-tune the pre-trained model on the labeled train split of the SPR_BENCH dataset.
- Use the dev split for hyperparameter tuning and model selection.

4. Evaluation:
- Evaluate the model on the test split using the chosen metric (either SWA or CWA).
- Compare the performance against the current SOTA.

## Risk Factors And Limitations

1. Complexity of Contrastive Learning: Designing effective positive and negative pairs for symbolic sequences may be challenging and could impact the quality of learned embeddings.
2. Generalization: Although contrastive learning aims to improve generalization, there is a risk that the learned embeddings may not transfer well to the specific SPR task.
3. Computational Resources: Pre-training a contrastive learning model on large symbolic sequences may require substantial computational resources, which could be a limitation for some academic labs.

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

