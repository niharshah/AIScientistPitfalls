## Name

symblic_polyrule_reasoning

## Title

Developing Robust Algorithms for Symbolic PolyRule Reasoning

## Short Hypothesis

By leveraging advanced machine learning techniques, we can develop algorithms capable of accurately classifying sequences governed by complex, poly-factor symbolic rules, outperforming existing rule-based classification methods.

## Related Work

Existing work in symbolic pattern recognition and rule-based classification often focuses on simpler, single-factor rules or combines numerical and symbolic methods. For example, the work by Gascuel et al. (1998) presents hybrid methods combining symbolic and numerical aspects but doesn't address the complexity of poly-factor rules. Similarly, Abdullah et al. (2003) explore rule-based knowledge discovery but focus on simpler rule structures. Our proposal aims to address this gap by focusing on multi-factor, logical AND-based rules in symbolic sequences.

## Abstract

This research aims to develop robust algorithms for Symbolic PolyRule Reasoning (SPR), a novel classification task involving sequences of abstract symbols governed by complex, poly-factor rules. These rules combine multiple atomic predicates, such as shape counts, color positions, parities, and order relations, to determine sequence acceptability. Existing symbolic pattern recognition methods often handle simpler rule structures and do not adequately address the intricacies of multi-factor logical rules. Our approach will involve designing and training machine learning models on the SPR_BENCH benchmark, comparing their performance against state-of-the-art baselines. The ultimate goal is to achieve superior classification accuracy, demonstrating the efficacy of our algorithms in handling complex symbolic rules.

## Experiments

- 1. **Model Development**: Develop various machine learning models (e.g., neural networks, decision trees) tailored to handle the poly-factor rules of the SPR task.
- 2. **Training and Tuning**: Train the models on the SPR_BENCH training set and tune them using the development set.
- 3. **Evaluation**: Evaluate the models on the SPR_BENCH test set, comparing the accuracy against the SOTA baseline of 70%.
- 4. **Ablation Studies**: Conduct ablation studies to understand the contribution of different model components in handling specific types of atomic predicates.
- 5. **Cross-Validation**: Perform cross-validation to ensure the generalizability of the models across different rule complexities and sequence lengths.

## Risk Factors And Limitations

1. **Complexity of Rules**: The complexity of poly-factor rules may pose a significant challenge, potentially requiring sophisticated model architectures. 2. **Generalization**: Ensuring that the models generalize well across different types of rules and sequence lengths may be difficult. 3. **Data Representation**: Effectively representing the symbolic sequences and their associated rules could be challenging, impacting model performance.

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

