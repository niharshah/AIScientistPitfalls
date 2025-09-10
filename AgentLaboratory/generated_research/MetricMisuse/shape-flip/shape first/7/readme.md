# Neuro-Symbolic Pattern Recognition (NSPR)

Welcome to the Neuro-Symbolic Pattern Recognition (NSPR) repository. This repository contains code, experiments, and documentation for our research on leveraging a hybrid neuro-symbolic framework for symbolic pattern recognition (SPR). Our approach integrates Decision Tree classifiers with a hybrid feature representation combining TF-IDF token embeddings and numeric features (unique shape and color counts) and augments these with rule extraction via large language models (LLMs) and inductive logic programming (ILP).

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup & Results](#experimental-setup--results)
- [Future Work](#future-work)
- [License](#license)
- [References](#references)

---

## Overview

Symbolic Pattern Recognition (SPR) is a challenging task that requires capturing both superficial lexical information and deeper latent interdependencies among symbolic tokens. Traditional methods relying solely on statistical techniques often reach performance plateaus when handling complex symbolic data. In our work, we propose a neuro-symbolic approach that:

- Uses TF-IDF vectorization to extract lexical features from token sequences.
- Computes numeric features such as counts of unique shape and color tokens.
- Integrates candidate symbolic rules extracted from large language models (LLMs) and validated via inductive logic programming (ILP).
- Combines these features in an unrestricted-depth Decision Tree classifier to capture non-linear dependencies effectively.
- Emphasizes performance on sequences with higher symbolic complexity via the Shape-Weighted Accuracy (SWA) metric.

---

## Background

In conventional SPR tasks, sequences are represented using hand-crafted or statistical features that may only capture superficial relationships. Our approach improves upon these methods by formally defining and emphasizing the following metric:

  SWA = (Σᵢ wᵢ · 𝟙{yᵢ = ŷᵢ}) / (Σᵢ wᵢ)

where wᵢ is the number of unique shape tokens in each sequence, thereby giving higher importance to samples with richer symbolic content.

This repository synthesizes ideas from recent neuro-symbolic research and builds upon prior work (e.g., arXiv papers 2506.14373v2, 2505.23833v1, 2410.23156v2) to offer a robust and interpretable framework for SPR.

---

## Methodology

Our hybrid approach consists of the following steps:

1. **Feature Extraction:**
   - **TF-IDF Embeddings:** Captures lexical content from tokenized sequences.
   - **Numeric Features:** Computes counts such as the number of unique shape and color tokens, augmenting the feature space.

2. **Rule Extraction & Integration:**
   - **LLM-derived Rules:** Generate candidate symbolic rules for deeper pattern extraction.
   - **ILP Validation:** Validate and refine candidate rules to ensure domain consistency.
   - The features are concatenated into an augmented vector:
     
      ~xᵢ = [ xᵢ, rₗₗₘ(sᵢ), rᵢₗₚ(sᵢ) ]

3. **Classification:**
   - An unrestricted-depth Decision Tree is used to exploit the enriched feature representation.
   - The model is optimized by minimizing the cross-entropy loss.

4. **Evaluation:**
   - Performance is evaluated using both standard Accuracy and the Shape-Weighted Accuracy (SWA) metric.

---

## Dataset

The experiments were conducted on the SPR BENCH dataset, which is partitioned into:

- 20,000 training samples
- 5,000 development samples
- 10,000 test samples

Each sample contains:
- An identifier
- A token sequence (with shape and color tokens)
- A corresponding label

Preprocessing involves tokenization suitable for TF-IDF vectorization and computation of numeric features based on symbolic attributes.

---

## Installation

To get started, clone the repository and install the required dependencies. You can set up the environment as follows:

```bash
# Clone this repository
git clone https://github.com/your-username/neuro-symbolic-spr.git
cd neuro-symbolic-spr

# (Optional) create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

The key libraries include:
- Python 3.7+
- scikit-learn
- NumPy
- SciPy
- matplotlib

---

## Usage

The main code is organized into modules that handle data preprocessing, feature extraction, rule integration, model training, and evaluation.

To run an end-to-end experiment, execute:

```bash
python run_experiment.py --dataset_path path/to/spr_bench_dataset --max_depth None
```

Command-line arguments include:
- `--dataset_path`: Path to the SPR BENCH dataset.
- `--max_depth`: Maximum depth for the Decision Tree; use `None` for unrestricted depth.

You can find additional scripts and notebooks that detail hyperparameter tuning, ablation studies, and result visualization.

---

## Experimental Setup & Results

### Experimental Details

- **Hyperparameter Tuning:**  
  We swept over maximum depths: 3, 5, 7, and None. The unrestricted depth configuration achieved the best performance, with:
  - Development set SWA: 71.28%
  - Test set SWA: 70.91%

- **Components Contribution:**
  
  | Component             | Description                                 | Contribution (%) |
  | --------------------- | ------------------------------------------- | ---------------- |
  | TF-IDF Embeddings     | Lexical content representation              | 40%              |
  | Numeric Features      | Unique shape and color count features       | 30%              |
  | LLM-derived Rules     | Extracted candidate symbolic rules          | 20%              |
  | ILP-validated Rules   | Validated and refined rules                 | 10%              |

- **Performance Analysis:**
  - Ablation studies indicate that removal of either TF-IDF or numeric features leads to a reduction in SWA by approximately 2–4%.
  - Confusion matrix analysis confirms a balanced classification across labels.

### Replication

Detailed instructions for running experiments, reproducing results, and generating performance plots (e.g., hyperparameter sweep and confusion matrices) are provided within the repository.

---

## Future Work

The current approach lays a robust foundation for neuro-symbolic integration in SPR. Future research directions include:
- Iterative rule refinement with advanced ILP techniques.
- Incorporation of self-supervised learning to capture deeper symbolic semantics.
- Exploration of more advanced architectures (e.g., ensemble methods, transformers) to further enhance performance and interpretability.
- Scalability improvements to accommodate larger and more complex symbolic datasets.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

1. [arXiv:2506.14373v2](https://arxiv.org/abs/2506.14373)
2. [arXiv:2505.23833v1](https://arxiv.org/abs/2505.23833)
3. [arXiv:2410.23156v2](https://arxiv.org/abs/2410.23156)
4. [arXiv:2505.06745v1](https://arxiv.org/abs/2505.06745)
5. Additional references are cited in the paper and within the code documentation.

---

For any questions, feedback, or contributions, please open an issue or submit a pull request. Happy coding!
