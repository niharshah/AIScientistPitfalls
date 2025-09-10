## Name

interpretable_neural_rule_learning

## Title

Interpretable Neural Rule Learning for Synthetic PolyRule Reasoning

## Short Hypothesis

Can we design an interpretable neural network model that can learn and explicitly represent the underlying poly-factor rules governing the Synthetic PolyRule Reasoning (SPR) task, enhancing both performance and interpretability?

## Related Work

1. Neural Rule Learning: Neural Logic Machines (Dong et al., 2018) and RL-Net (Dierckx et al., 2023) focus on learning logical rules but often lack interpretability and explicit rule representation. 2. Symbolic Reasoning Models: Deep Concept Reasoner (Barbiero et al., 2023) builds syntactic rule structures but still relies on high-dimensional concept embeddings. 3. Interpretable AI: Existing methods like LIME and SHAP offer post-hoc explanations but do not inherently learn interpretable rules. Our proposal aims to develop a neural network model that inherently learns and represents poly-factor rules in an interpretable manner, addressing the performance-interpretability trade-off.

## Abstract

The Synthetic PolyRule Reasoning (SPR) task involves classifying symbolic sequences based on latent poly-factor rules. Current approaches in neural rule learning and symbolic reasoning either lack interpretability or are domain-specific. This proposal aims to develop an interpretable neural network model that learns and explicitly represents the underlying poly-factor rules governing the SPR task. By integrating rule-based learning with neural networks, we aim to create a model that not only achieves high classification accuracy but also provides interpretable rule representations. Our approach will be evaluated on the SPR_BENCH benchmark from HuggingFace, aiming to surpass the state-of-the-art accuracy of 80.0% while providing clear rule explanations for each classification decision.

## Experiments

- {'Description': 'Model Design', 'Details': 'Develop a neural network model that incorporates a rule-based layer designed to learn and represent poly-factor rules. This layer will output explicit rules in human-readable format.'}
- {'Description': 'Training and Evaluation', 'Details': 'Train the model on the Train split of the SPR_BENCH benchmark. Tune the model on the Dev split. Evaluate final accuracy on the Test split and compare against the SOTA accuracy of 80.0%.'}
- {'Description': 'Interpretability Analysis', 'Details': 'Extract and present the learned rules for a subset of sequences. Conduct user studies to evaluate the interpretability of the rules.'}
- {'Description': 'Ablation Studies', 'Details': 'Compare performance with and without the rule-based layer. Evaluate the impact of different rule complexities on model performance.'}

## Risk Factors And Limitations

- Complexity of Rule Learning: The model might struggle to learn highly complex rules, leading to reduced performance.
- Interpretability Trade-offs: Balancing performance and interpretability may be challenging, potentially leading to trade-offs.
- Generalization: The model's ability to generalize to unseen rules and sequences may be limited.

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

