# Robust PolyRuleNet for Synthetic PolyRule Reasoning

Welcome to the Robust PolyRuleNet repository! This project presents a two-stage neuro-symbolic model aimed at enhancing the interpretability and performance of synthetic poly-rule reasoning tasks. The approach explicitly decouples the transformation of continuous token embeddings into discrete symbolic representations from the subsequent rule induction process, thereby providing increased transparency while achieving competitive performance on complex rule-based datasets.

---

## Overview

**Robust PolyRuleNet** addresses the task of Synthetic PolyRule Reasoning (SPR) by:
- Transforming input tokens into discrete, one-hot symbolic representations using a differentiable discretization method (Gumbel-Softmax).
- Aggregating these discrete symbols via mean pooling and processing them with a lightweight multilayer perceptron (MLP) to provide binary predictions on whether the input adheres to hidden poly-factor rules.
- Focusing on predicates such as shape-count, color-position, parity, and order to simulate challenging reasoning tasks that require both robust performance and strong interpretability.

The project demonstrates that explicit symbolic tokenization not only aids in understanding the internal decision process but also delivers marginal improvements over existing baselines (e.g., achieving a Shape-Weighted Accuracy (SWA) of 60.58% on the test set).

---

## Repository Structure

- **/paper/**  
  Contains the LaTeX source for the paper “Research Report: Robust PolyRuleNet for Synthetic PolyRule Reasoning” that details our methodology, experimental setup, and findings.

- **/code/**  
  Implements the Robust PolyRuleNet model:
  - **model.py**: Defines the two-stage architecture, including the Gumbel-Softmax based discretization and the MLP-based rule induction.
  - **train.py**: Scripts to train the model on the synthetic SPR dataset.
  - **utils.py**: Utilities for data generation, evaluation metrics, and configuration settings.

- **/experiments/**  
  Scripts and notebooks for reproducing the experiments reported in the paper, including ablation studies and sensitivity analysis on the temperature parameter.

- **/data/**  
  Synthetic datasets generated for training, validation, and testing. (Note: Data generation scripts are also available in the code directory.)

- **README.md**  
  This file, providing a high-level overview and instructions for using the repository.

---

## Features

- **Differentiable Discretization with Gumbel-Softmax:**  
  Converts continuous token embeddings into discrete one-hot vectors while preserving essential relational information.

- **Neuro-Symbolic Integration:**  
  Combines the strengths of neural networks and explicit symbolic reasoning to validate poly-factor rules over sequences that include predicates like shape-count, color-position, parity, and order.

- **Interpretability:**  
  By enforcing explicit symbolic tokenization, the model’s internal decision-making is more transparent and easier to audit compared to traditional end-to-end architectures.

- **Reproducible Experiments:**  
  Detailed experimental setups, including training configurations (batch size, learning rate, training epochs) and evaluation protocols (binary accuracy, shape-weighted accuracy), are provided to facilitate reproducibility.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/RobustPolyRuleNet.git
   cd RobustPolyRuleNet
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** The requirements file contains all necessary libraries such as PyTorch, NumPy, and others.

---

## Usage

### Training the Model

To train Robust PolyRuleNet on the provided synthetic dataset, run:

```bash
python code/train.py --config configs/train_config.yaml
```

- The configuration file allows you to set parameters such as learning rate, batch size, number of epochs, and the Gumbel-Softmax temperature parameter.

### Evaluating the Model

After training, evaluate the model on the development or test set using:

```bash
python code/evaluate.py --config configs/eval_config.yaml
```

Evaluation metrics include binary accuracy and Shape-Weighted Accuracy (SWA).

### Data Generation

If you wish to generate your own synthetic SPR datasets, use the provided data generation script:

```bash
python code/data_generator.py --output data/synthetic_dataset.json --num_samples 20000
```

---

## Experimental Details

- **Architecture:**  
  The model first projects 32-dimensional token embeddings into a 16-dimensional logit space corresponding to discrete symbols. The Gumbel-Softmax function is then applied with a controllable temperature parameter.
  
- **Rule Induction Module:**  
  A lightweight MLP processes the mean-pooled symbolic representations, using two layers with ReLU activations, followed by a sigmoid output yielding binary predictions.

- **Training Setup:**  
  The model uses the Adam optimizer with a default learning rate of 1e-3, a batch size of 64, and a typical training schedule of 2 epochs. Detailed ablation studies have shown that the explicit symbolic tokenization stage is crucial for effective rule induction.

- **Results:**  
  The model achieved an initial training loss decrease from 0.6884 to 0.6611 over two epochs, a test set binary accuracy of ~60.28%, and a marginal improvement in SWA over the SPR_BENCH baseline.

For more in-depth details, please refer to the paper located in the `/paper/` directory.

---

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with detailed descriptions and tests.

Please ensure that your code adheres to the existing style and includes appropriate tests.

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite our work as follows:

```
@misc{RobustPolyRuleNet2023,
    title={Research Report: Robust PolyRuleNet for Synthetic PolyRule Reasoning},
    author={Agent Laboratory},
    year={2023},
    note={Available at https://github.com/yourusername/RobustPolyRuleNet}
}
```

---

## Contact

For any questions or suggestions, feel free to open an issue in the repository or contact [your.email@example.com](mailto:your.email@example.com).

Happy coding and exploring neuro-symbolic reasoning!

