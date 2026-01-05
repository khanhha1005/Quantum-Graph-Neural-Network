# Quantum GCN Model Results

## Model Configuration

### Current Configuration
- **Quantum layer depths**: `q_depths=[1]`
- **Number of layers (L)**: 1 QGCNConv layer
- **Number of qubits**: 8 (fixed for input_dims > 4)
- **Output dimensions**: 1 (binary classification)

## Parameter Count Results

### Current Configuration (q_depths=[1], L=1 layer)

| Classifier Type | Number of Parameters |
|----------------|---------------------|
| **MPS** | **2,039 parameters** |
| **TTN** | **2,039 parameters** |
| **Linear** | **2,032 parameters** |

## Parameter Breakdown

### For MPS/TTN Classifiers (2,039 parameters)

| Component | Parameters | Details |
|-----------|------------|---------|
| **Layer 1 - Feature reduction** | 2,000 | Linear transformation: input_dims × n_qubits |
| **Layer 1 - Quantum circuit** | 16 | Variational layers: q_depth × n_qubits × 2 |
| **Layer 1 - Bias** | 8 | Learnable bias vector |
| **QGCNConv Total** | **2,024** | |
| **Classifier (MPS/TTN)** | 15 | Quantum classifier parameters |
| **TOTAL** | **2,039** | |

### For Linear Classifier (2,032 parameters)

| Component | Parameters | Details |
|-----------|------------|---------|
| **Layer 1 - Feature reduction** | 2,000 | Linear transformation: input_dims × n_qubits |
| **Layer 1 - Quantum circuit** | 16 | Variational layers: q_depth × n_qubits × 2 |
| **Layer 1 - Bias** | 8 | Learnable bias vector |
| **QGCNConv Total** | **2,024** | |
| **Classifier (Linear)** | 8 | Linear layer: n_qubits × output_dims |
| **TOTAL** | **2,032** | |

## Key Characteristics

### Model Architecture
- **Input**: Node features with variable dimensions (e.g., 250)
- **Feature reduction**: Maps input features to qubit space (8 qubits)
- **Quantum processing**: Variational quantum circuits process node embeddings
- **Graph aggregation**: Global mean pooling for graph-level representation
- **Classification**: Quantum or classical classifier for binary prediction

### Quantum Components
- **Number of qubits**: 8 (power of 2, compatible with TTN)
- **Quantum circuit depth**: 1 layer per QGCNConv
- **Quantum gates**: RY and RZ rotations, CZ entangling gates
- **Measurement**: Expectation values (Pauli-Z)

### Parameter Efficiency
- **Total parameters**: ~2,000-2,040 range
- **No traditional GCN weight matrices**: Eliminated and replaced by quantum circuits
- **All parameters are trainable**: No parameter tying or sharing
- **Compact model**: Efficient parameter usage for quantum-enhanced GNN

## Comparison with Other Configurations

### Maximum Configuration (q_depths=[2, 2], L=2 layers)
- **MPS/TTN**: 2,095 parameters
- **Linear**: 2,088 parameters

The current configuration (L=1) provides a good balance between model capacity and parameter efficiency, achieving approximately **2K parameters** as claimed.

## Notes

- Parameters are counted for **trainable weights only**
- Feature reduction matrices have **no bias** (bias=False)
- Quantum circuits replace traditional linear transformations
- Message passing uses normalized adjacency (no learnable weights)
- All parameters are independent (no parameter sharing)

## Verification

The parameter counts have been verified through:
- Direct code inspection of model architecture
- Parameter counting script (`calculate_model_params.py`)
- Model initialization and parameter enumeration

For detailed parameter counting methodology, see `PARAMETER_COUNTING_METHODOLOGY.md`.

