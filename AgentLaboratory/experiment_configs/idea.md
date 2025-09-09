## Name

polyrule_net

## Title

PolyRuleNet: Hierarchical Rule Extraction for Synthetic PolyRule Reasoning

## Short Hypothesis

Decomposing complex reasoning into interpretable, atomic predicate evaluations enables a hierarchical neural-symbolic architecture (PolyRuleNet) to outperform black-box models on the Synthetic PolyRule Reasoning task.

## Related Work

Most existing approaches treat symbolic pattern recognition as an end-to-end black-box classification problem, relying on deep networks to implicitly learn complex rules. Prior literature in fuzzy logic and traditional pattern recognition offers interpretability but lacks scalability and integration with modern deep learning advances. PolyRuleNet distinctly combines deep feature extraction with dedicated modules for evaluating atomic predicates (shape-count, color-position, parity, and order), providing both competitive performance and interpretability under a unified framework.

## Abstract

We introduce PolyRuleNet, a novel hierarchical neural-symbolic model designed to tackle Synthetic PolyRule Reasoning (SPR), a challenging classification task where hidden poly-factor rules govern the acceptance or rejection of symbolic sequences. Unlike conventional methods that rely on end-to-end training, our model explicitly decomposes the classification task into several predicate evaluation modules aligned with underlying rule categories such as shape-count, color-position, parity, and order. First, a deep feature extractor processes the symbolic sequence and generates robust representations. Then, parallel predicate modules assess these representations based on defined logical conditions. The final decision is derived by aggregating module outputs using an enforced logical AND operation mirroring the rule structure. We evaluate our model on the SPR_BENCH benchmark, aiming to exceed the current SOTA accuracy of 88.9% while enhancing interpretability by directly associating model components with rule predicates. A series of ablation studies and generalization experiments across variants of sequence lengths and vocabulary sizes provide insights into the contributions of each module and the model’s robustness. Our results suggest that PolyRuleNet successfully bridges neural network efficiency and symbolic reasoning clarity, paving the way for more interpretable automated reasoning systems.

## Experiments

1. Baseline Comparison: Train PolyRuleNet on the SPR_BENCH training set and evaluate on the test set, comparing accuracy, F1 score, and inference time against baseline deep learning models. 2. Ablation Study: Sequentially disable each predicate module (shape-count, color-position, parity, order) to quantify its individual contribution to overall accuracy. 3. Generalization & Robustness: Test the model on altered datasets varying in sequence length and vocabulary size to assess performance adaptability. 4. Interpretability Analysis: Visualize and analyze intermediate outputs of predicate modules to verify if they align with expected symbolic rules.

## Risk Factors And Limitations

1. Increased model complexity may lead to overfitting, especially on limited training data. 2. Misalignment between learned predicates and true symbolic semantics could obscure interpretability benefits. 3. Balancing modular design with end-to-end optimization poses challenges that may require careful tuning. 4. In scenarios where predicate interdependencies are strong, a strictly modular approach may lose some performance compared to fully integrated models.

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
 ├─ train.csv   (2 000 rows)
 ├─ dev.csv     (  500 rows)
 └─ test.csv    (1 000 rows)

Each CSV has header:  id,sequence,label
────────────────────────────────────────────────────────
$ pip install datasets   # once
$ python read_spr_bench.py  --root SPR_BENCH
"""
import argparse
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


def main():

    ## Absolute path of the datasets
    DATA_PATH = pathlib.Path('/home/zxl240011/AI-Scientist-v2/SPR_BENCH/')
    spr_bench = load_spr_bench(DATA_PATH)

    print("Benchmarks split:", spr_bench.keys())

    # Demo: show first example from IJSJF‑train
    ex = spr_bench["train"][0]
    print("\nExample row:")
    print(ex)          


if __name__ == "__main__":
    main()

```

