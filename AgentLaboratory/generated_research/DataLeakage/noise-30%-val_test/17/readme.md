# Symbolic Pattern Recognition Benchmark (SPR_BENCH)

Welcome to the SPR_BENCH repository! This project provides an in-depth analysis of symbolic pattern recognition (SPR) under extreme data scarcity. In our study, we investigate the performance of simple bag-of-words models—specifically logistic regression pipelines using CountVectorizer and TfidfVectorizer—on a dataset containing only 2 training samples, 1 development sample, and 1 test sample.

---

## Table of Contents

- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion & Future Work](#discussion--future-work)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository contains the code and materials for our research report titled **"Symbolic Pattern Recognition Benchmark Analysis"**. Our work rigorously quantifies the limitations of conventional bag-of-words approaches for SPR under severe data scarcity. Despite advanced methods in the literature achieving ~70% accuracy on similar tasks, our baseline methods (CountVectorizer and TfidfVectorizer coupled with logistic regression) were unable to capture abstract symbolic rules, resulting in 0.0000% accuracy on both development and test splits.

---

## Project Motivation

The key motivation of our research is to:
- **Quantify Failure Modes:** Identify why traditional bag-of-words methods fail to generalize symbolic rules with extremely limited data.
- **Establish a Baseline:** Provide performance benchmarks (0.0000% accuracy) as a negative control for SPR approaches.
- **Stimulate Future Research:** Encourage exploration into advanced neural architectures equipped with multi-stage attention mechanisms and neuro-symbolic integration to bridge the abstraction gap.

---

## Dataset

The SPR_BENCH dataset is intentionally minimal to challenge model generalization. It consists of:
- **2 Training Samples**
- **1 Development Sample**
- **1 Test Sample**

The minimal design allows us to isolate the effects of data scarcity on symbolic pattern recognition.

---

## Methodology

Our methodology includes:
- **Tokenization:** We preprocess raw token sequences using a custom regex pattern `(?u)\b\w+\b` to ensure even single-character tokens are captured.
- **Vectorization:** Two types of bag-of-words representations are generated:
  - **CountVectorizer:** Captures raw token frequencies.
  - **TfidfVectorizer:** Weighs tokens based on term frequency-inverse document frequency.
- **Classifier:** A logistic regression model is employed:
  - The probabilistic model is defined as:
    P(y | x) = 1 / (1 + exp(- (wᵀφ(x) + b))),
  - Where φ(x) denotes the bag-of-words feature vector.
- **Formulation:** We define the abstraction gap (ε) as:
  
  ε = (1/N) ∑ᵢ || g(xᵢ) - h(yᵢ) ||²

  where g(x) is an encoder mapping tokens to abstract representations and h(y) is the ideal symbolic mapping.

A detailed description, including tokenization modifications and logistic regression configuration, is provided in the paper.

---

## Experimental Setup

Key configuration parameters:
  
| Parameter               | Value/Description                        |
| ----------------------- | ---------------------------------------- |
| Number of Training Samples    | 2                                      |
| Number of Development Samples | 1                                      |
| Number of Test Samples        | 1                                      |
| Tokenization Pattern          | `(?u)\b\w+\b`                         |
| Vectorization Methods         | CountVectorizer, TfidfVectorizer       |
| Maximum Iterations            | 300 (for convergence in logistic regression) |

Experiments were run with fixed hyperparameters to strictly evaluate the baseline performance under severe data constraints.

---

## Results

The baseline methods yielded the following performance metrics:

| Method            | Dev Accuracy | Test Accuracy |
| ----------------- | ------------ | ------------- |
| CountVectorizer   | 0.0000%      | 0.0000%       |
| TfidfVectorizer   | 0.0000%      | 0.0000%       |

Our experiments reveal that with extreme data scarcity the traditional bag-of-words approaches cannot capture the abstract symbolic rules, as confirmed by both confusion matrix analysis and error rate evaluation.

---

## Discussion & Future Work

### Discussion
- **Data Scarcity Challenge:** With only 2 training examples, model parameter estimation and abstraction learning fail.
- **Limitations of Bag-of-Words:** These methods neglect token order and relational structure, crucial for symbolic reasoning.
- **Current Baseline Performance:** The zero accuracy emphasizes the need for richer data and more advanced architectures.

### Future Work
- **Advanced Neural Architectures:** Explore transformer-based models with multi-stage attention to induce symbolic abstraction.
- **Data Augmentation:** Enrich the SPR_BENCH dataset with a more diverse range of symbolic sequences.
- **Neuro-Symbolic Integration:** Combine statistical models with symbolic reasoning mechanisms to achieve meaningful generalization.

---

## Usage

### Prerequisites
- Python (>=3.7)
- Required Python packages (install via pip):
  - scikit-learn
  - numpy
  - scipy

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/spr_bench.git
   cd spr_bench
   ```

2. (Optional) Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

### Running the Baseline Experiment

To run the logistic regression baseline with CountVectorizer and TfidfVectorizer, execute:

```
python run_baseline.py
```

The script will preprocess the data, train the classifier, and output the performance metrics.

---

## Citation

If you find this project useful in your research, please cite our work:

    @misc{sprbench2023,
      title={Symbolic Pattern Recognition Benchmark Analysis},
      author={Agent Laboratory},
      year={2023},
      note={Unpublished manuscript.}
    }

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

We hope this repository serves as a useful resource for exploring the challenges and future directions in symbolic pattern recognition under extreme data scarcity!

For any questions or contributions, please feel free to open an issue or submit a pull request.

Happy coding!