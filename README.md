# Quantum-aided Graph Neural Networks for Enhancing Smart Contract Vulnerability Detection

This repository implements quantum-enhanced Graph Neural Networks (QGNNs) for detecting vulnerabilities in smart contracts. The approach leverages quantum computing principles to enhance graph neural network architectures for improved vulnerability detection performance.

## Overview

This work presents a quantum-classical hybrid approach that combines:
- **Quantum Graph Convolution Layers (QGCNConv)**: Quantum-enhanced node embedding using variational quantum circuits
- **Quantum Classifiers**: Matrix Product State (MPS) and Tree Tensor Network (TTN) quantum classifiers

## Model Architecture

### Quantum Graph Convolutional Network (QGCN)

The QGCN architecture consists of three main components:

#### 1. Quantum Graph Convolution Layers (QGCNConv)

Each QGCNConv layer performs:
- **Feature Reduction**: Linear transformation to map input features to qubit space (if `in_channels ≠ n_qubits`)
- **Quantum Node Embedding**: Variational quantum circuit (VQC) processes each node feature vector
  - **Qubits**: 2 , 4 or 8 qubits (power of 2, determined by input dimensions)
  - **Quantum Layers**: Configurable depth (default: 1 layer)
  - **Circuit Structure**: 
    - Angle embedding (Y-rotation) for feature encoding
    - Variational layers with RY and RZ rotations
    - Entangling gates (CZ) in ring topology
    - Expectation value measurements (Pauli-Z)
- **Graph Message Passing**: Aggregates quantum embeddings via normalized adjacency matrix
- **Bias Addition**: Learnable bias vector

**Parameters per QGCNConv layer:**
- Feature reduction: `in_channels × n_qubits` (if needed)
- Quantum circuit: `n_layers × n_qubits × 2` (RY + RZ rotations per qubit per layer)
- Bias: `n_qubits`

#### 2. Graph Pooling

- **Global Mean Pooling**: Aggregates node embeddings to graph-level representation

#### 3. Quantum Classifiers

Three classifier options:

**a) Matrix Product State (MPS) Classifier**
- **Architecture**: Sequential qubit entanglement with CZ gates
- **Parameters**: `2 × n_qubits - 1` trainable quantum weights
- **Measurement**: Expectation values on specified qubits (typically last qubit for binary classification)
- **Circuit**: Angle embedding → Variational layers (RY rotations + CZ gates) → Measurement

**b) Tree Tensor Network (TTN) Classifier**
- **Architecture**: Hierarchical tree structure (requires power-of-2 qubits)
- **Parameters**: `2 × (n_qubits - 1) + 1` trainable quantum weights
- **Measurement**: Single expectation value on final qubit
- **Circuit**: Angle embedding → Tree-structured variational layers → Measurement

**c) Linear Classifier (Baseline)**
- **Architecture**: Classical linear layer
- **Parameters**: `n_qubits × output_dims`
## Dataset

The models are trained on smart contract vulnerability detection datasets:
- **Reentrancy**: Detecting reentrancy vulnerabilities
- **Integer Overflow**: Detecting integer overflow vulnerabilities  
- **Timestamp Dependency**: Detecting timestamp-dependent vulnerabilities

### Data Preparation

The training data in `train_data/` is obtained by running the **graph construction** and **graph normalization** steps from the [GNNSCVulDetector](https://github.com/Messi-Q/GNNSCVulDetector) repository. 

To prepare the training data:
1. Clone the [GNNSCVulDetector](https://github.com/Messi-Q/GNNSCVulDetector) repository
2. Follow their instructions to run graph construction and graph normalization on your smart contract dataset
3. The output JSON files should be placed in the `train_data/` directory with the following structure:
   ```
   train_data/
   ├── reentrancy/
   │   ├── train.json
   │   └── valid.json
   ├── integeroverflow/
   │   ├── train.json
   │   └── valid.json
   └── timestamp/
       ├── train.json
       └── valid.json
   ```

### Data Format

Each dataset contains:
- **Graph-structured code representations**: Nodes represent code elements (statements, expressions, etc.), edges represent relationships between code elements
- **Binary labels**: Each graph is labeled as vulnerable (1) or non-vulnerable (0)
- **JSON format**: Each JSON file contains a list of graphs with:
  - `node_features`: Feature vectors for each node
  - `graph`: Edge list in format `[source, edge_type, target]`
  - `targets`: Binary label (0 or 1)

## Installation

### Requirements

```bash
pip install torch torch-geometric
pip install pennylane pennylane-lightning
pip install scikit-learn tensorboard tqdm
```
## Usage

### Training Quantum Models

Run the training script to train all quantum models:

```bash
python train_quantum_models.py
```

This will:
1. Train QGCN models with MPS, TTN, and Linear classifiers
2. Train on all three vulnerability types (reentrancy, integeroverflow, timestamp)
3. Save models, training history, and configurations
4. Generate TensorBoard logs for visualization

### Configuration

Edit the `config` dictionary in `train_quantum_models.py` to adjust hyperparameters:

```python
config = {
    'batch_size': 8,              # Batch size for training
    'learning_rate': 0.001,         # Learning rate for Adam optimizer
    'max_epochs': 100,              # Maximum training epochs
    'patience': 15,                 # Early stopping patience
    'min_delta': 0.0,               # Minimum improvement threshold
    'q_depths': [1],                # Quantum layer depths per QGCNConv
    'validate_every': 3,            # Validation frequency (epochs)
    'max_nodes_per_batch': 5000,     # Maximum nodes per batch (filtering)
}
```

### Model Selection

To train specific models, modify the classifier types:

```python
classifier_types = ['MPS', 'TTN', 'Linear']  # All classifiers
```

## TensorBoard Setup and Visualization

### 1. Training Logs

TensorBoard logs are automatically saved during training to:
```
training_results/[vulnerability_type]_[classifier_type]_[timestamp]/logs/
```

### 2. Launching TensorBoard

```bash
tensorboard --logdir=training_results
```
### 3. Accessing TensorBoard

Open your web browser and navigate to:
```
http://localhost:6006
```

### 4. Available Metrics

TensorBoard displays:
- **Loss**: Training and validation loss
- **Accuracy**: Classification accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1 Score**: F1 score
- **Best Validation F1**: Best F1 score achieved
- **Model Information**: Number of parameters, input dimensions

## Output Files

After training, each model generates:

- **`model.pt`**: Final model weights (last epoch)
- **`best_model.pt`**: Best model weights (highest validation F1)
- **`history.json`**: Training history (loss, accuracy, precision, recall, F1)
- **`config.json`**: Training configuration
- **`logs/`**: TensorBoard event files
