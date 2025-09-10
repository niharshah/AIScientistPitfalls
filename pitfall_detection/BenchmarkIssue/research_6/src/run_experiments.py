import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from datasets import load_dataset

# Dataset loading
datasets = {
    "DFWZN": load_dataset('csv', data_files='SPR_BENCH/DFWZN/train.csv')['train'],
    "JWAEU": load_dataset('csv', data_files='SPR_BENCH/JWAEU/train.csv')['train'],
    "GURSG": load_dataset('csv', data_files='SPR_BENCH/GURSG/train.csv')['train'],
    "QAVBE": load_dataset('csv', data_files='SPR_BENCH/QAVBE/train.csv')['train'],
    "IJSJF": load_dataset('csv', data_files='SPR_BENCH/IJSJF/train.csv')['train']
}

# Convert sequence to graph
def sequence_to_graph(sequence):
    G = nx.Graph()
    for i, token in enumerate(sequence):
        G.add_node(i, shape_color=token)
        if i > 0:
            G.add_edge(i - 1, i)
    return G

# Data processing: Convert characters to ASCII numerical
def process_data(sequence):
    try:
        return np.array([ord(char) for char in sequence if 0 <= ord(char) < 128])
    except ValueError:
        return np.zeros(len(sequence))

# Bayesian Network Classifier
def bayesian_network_classifier(X_train, y_train, X_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Algorithm Selection Mechanism
def algorithm_selection(y_pred_bayesian, y_pred_nn):
    # Directly use Bayesian predictions in this placeholder implementation
    return y_pred_bayesian

# Evaluation and Visualization
def evaluate_and_visualize(dataset, model_name):
    print(f"Running {model_name} on dataset...")
    
    # Prepare the data
    sequences = dataset['sequence']
    labels = np.array(dataset['label'])

    processed_data = [process_data(seq) for seq in sequences]
    max_length = max(map(len, processed_data))
    padded_data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in processed_data])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.25, random_state=42)
    
    # Run Bayesian classifier
    y_pred_bayesian = bayesian_network_classifier(X_train, y_train, X_test)
    
    # Placeholder for Fuzzy NN results using Bayesian output
    y_pred_nn = y_pred_bayesian
    
    # Select the best algorithm predictions
    selected_preds = algorithm_selection(y_pred_bayesian, y_pred_nn)
    accuracy = accuracy_score(y_test, selected_preds)
    
    print(f"Accuracy for {model_name}: {accuracy * 100:.2f}%\n")
    
    # Visualization
    plt.figure()
    plt.hist(y_test, bins='auto', alpha=0.7, label="True Labels")
    plt.hist(selected_preds, bins='auto', alpha=0.7, label="Predicted Labels (Selected)")
    plt.title(f"{model_name} - True vs Selected Predictions")
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.legend(loc='best')
    plt.savefig(f"{model_name}_Results.png")

# Run evaluation across all datasets
for name, dataset in datasets.items():
    evaluate_and_visualize(dataset, name)