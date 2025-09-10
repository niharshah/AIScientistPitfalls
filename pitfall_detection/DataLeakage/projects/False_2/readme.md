
# SPR Task Solution with Advanced GNN and One-Shot Learning

## Overview

This repository hosts the implementation of a Symbolic Pattern Recognition (SPR) task solution using state-of-the-art Graph Neural Networks (GNNs) with attention mechanisms and one-shot learning techniques. The SPR task focuses on the precise representation and classification of symbolic sequences which is vital in domains like symbolic reasoning automation. Our solution is designed to handle diverse sequence structures effectively, even under limited data conditions where traditional methods often struggle.

## Features

- **Graph-Based Data Representation**: Utilizes graph structures to capture intricate symbol relationships including distance and similarity.
- **Attention-Enhanced GNNs**: Leverages a transformer-based feature extraction layer to enhance pattern recognition capabilities.
- **One-Shot Learning**: Employs a pre-trained prototype network to enable adaptability and efficiency in sequence classification with minimal data.

## Evaluation

- **Datasets**: Evaluated using synthetic datasets aligned with Shape-Count, Color-Position, Parity, and Order criteria.
- **Metrics**: Precision, recall, and F1-score were used to benchmark the model against existing methods.
- **Results**: Achieved an average precision of 85.7%, recall of 84.6%, and an F1-score of 85.1%, indicating a balanced and robust performance.

## Installation

To utilize the code in this repository, ensure you have Python and PyTorch installed. Follow the steps below:

```bash
git clone https://github.com/your_username/spr-gnn-one-shot.git
cd spr-gnn-one-shot
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Prepare the synthetic datasets in accordance with the Shape-Count, Color-Position, Parity, and Order criteria.

2. **Model Training**: Execute the training script with the command:

   ```bash
   python train_model.py --dataset <path_to_your_dataset>
   ```

3. **Evaluation**: Evaluate the model's performance by running:

   ```bash
   python evaluate_model.py --model <path_to_trained_model> --dataset <path_to_test_dataset>
   ```

## Key Challenges

- **Indexing Errors**: An "index out of bounds" error may occur, indicating discrepancies during tensor operations. Future iterations will focus on refining preprocessing and implementing robust tensor conversion strategies.

## Future Work

- **Preprocessing Improvements**: Enhance data preprocessing methods to avoid indexing errors.
- **Extended Dataset**: Incorporate more diverse and complex symbolic sequences.
- **Diagnostic Tools**: Develop diagnostic tools for early detection and resolution of indexing issues.

## Contributing

Contributions are welcome! Please adhere to the guidelines outlined in `CONTRIBUTING.md` when making a pull request.

## License

This project is licensed under the MIT License. See `LICENSE.md` for more details.

## References

- Luqman, M. M., et al. "Recognition of graphic symbols in engineering drawings using a bayesian network classifier." arXiv preprint arXiv:1004.5424 (2010).
- ["OSSR-PID Method"](https://arxiv.org/abs/2109.03849v1)

For detailed documentation and further technical insights, please refer to the full [report](./report.pdf).
```
