# Quantum GCN Parameter Counting Methodology

## Answer to Question 1: Parameters for input_dim = 250

### Configuration
- **Input dimensions**: 250
- **Number of qubits**: 8 (fixed, determined by `min(input_dims, 8)` then rounded to power of 2)
- **Output dimensions**: 1 (binary classification)

### Results

#### Current Configuration (q_depths=[1], L=1 layer)
- **MPS classifier**: **2,039 parameters**
- **TTN classifier**: **2,039 parameters**
- **Linear classifier**: **2,032 parameters**

#### Maximum Configuration (q_depths=[2, 2], L=2 layers)
- **MPS classifier**: **2,095 parameters**
- **TTN classifier**: **2,095 parameters**
- **Linear classifier**: **2,088 parameters**

---

## Answer to Question 2: Parameter Counting Methodology Specification

### Addressing the Concern

> "The parameter counting methodology is not fully specified (e.g., number of layers L, feature dimensionalities, whether GCN weight matrices are eliminated or tied), making it difficult to verify the 2K total parameter claim."

### Detailed Clarifications

#### 1. Number of Layers (L)

**Definition**: `L = len(q_depths)` = number of QGCNConv layers

- **Current config**: `L = 1` (q_depths=[1])
- **Maximum config**: `L = 2` (q_depths=[2, 2])

**Each QGCNConv layer contains**:
- Feature reduction matrix (if `input_dims ≠ n_qubits`)
- Quantum variational circuit
- Bias vector

**Code reference**: `src/models/Quantum_GCN.py` lines 31-35

#### 2. Feature Dimensionalities

**Flow through the network**:
- **Input**: `input_dims` (e.g., 250 from node features)
- **After first layer**: Reduced to `n_qubits` (8)
- **Subsequent layers**: `n_qubits → n_qubits` (no reduction needed)
- **Output**: `n_qubits` (8) → `output_dims` (1) via classifier

**Feature reduction**:
- Only needed when `input_dims ≠ n_qubits`
- First layer: `input_dims × n_qubits` parameters
- Subsequent layers: No reduction needed (already `n_qubits`)

**Code reference**: `src/models/QCN_Layers/QGCNConv.py` lines 42-45

#### 3. GCN Weight Matrices

**Critical clarification**: 
- ❌ **NO traditional GCN weight matrices** (W matrices are **ELIMINATED**)
- ✅ **Quantum circuits replace** the linear transformation
- ✅ Message passing uses **normalized adjacency** (no learnable weights)
- ✅ Only **bias vectors** are learnable in message passing step

**Traditional GCN**: `H' = σ(D^(-1/2) A D^(-1/2) H W)`
- `W` is a learnable weight matrix

**Quantum GCN**: `H' = D^(-1/2) A D^(-1/2) Q(H) + b`
- `Q(H)` is quantum circuit (replaces `H W`)
- `b` is learnable bias vector
- No weight matrix `W`

**Code reference**: `src/models/QCN_Layers/QGCNConv.py` lines 56-85

#### 4. Parameter Tying

**Parameters are NOT tied across layers**:
- Each QGCNConv layer has **independent parameters**
- Each layer has its own:
  - Feature reduction matrix (if needed)
  - Quantum circuit weights
  - Bias vector

**No parameter sharing** between layers.

#### 5. Parameter Counting Formula

For `L` QGCNConv layers:

```
Total = Σ[layer=1 to L] {
    (input_dims[layer] × n_qubits) if input_dims[layer] ≠ n_qubits else 0
    + (q_depth[layer] × n_qubits × 2)  // Quantum circuit: RY + RZ per qubit per layer
    + n_qubits  // Bias vector
}
+ classifier_params
```

**Where**:
- `input_dims[layer] = input_dims` if `layer == 1`, else `n_qubits`
- `q_depth[layer]` = quantum circuit depth for layer `layer`
- `classifier_params` depends on classifier type:
  - MPS: `2 × n_qubits - 1`
  - TTN: `2 × (n_qubits - 1) + 1`
  - Linear: `n_qubits × output_dims`

#### 6. Detailed Breakdown for input_dim = 250

**Current config (L=1, q_depths=[1])**:

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Layer 1 - Feature reduction | 2,000 | 250 × 8 |
| Layer 1 - Quantum circuit | 16 | 1 × 8 × 2 |
| Layer 1 - Bias | 8 | 8 |
| **QGCNConv total** | **2,024** | |
| Classifier (MPS/TTN) | 15 | 2 × 8 - 1 |
| **TOTAL** | **2,039** | |

**Maximum config (L=2, q_depths=[2, 2])**:

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Layer 1 - Feature reduction | 2,000 | 250 × 8 |
| Layer 1 - Quantum circuit | 32 | 2 × 8 × 2 |
| Layer 1 - Bias | 8 | 8 |
| Layer 2 - Feature reduction | 0 | Not needed |
| Layer 2 - Quantum circuit | 32 | 2 × 8 × 2 |
| Layer 2 - Bias | 8 | 8 |
| **QGCNConv total** | **2,080** | |
| Classifier (MPS/TTN) | 15 | 2 × 8 - 1 |
| **TOTAL** | **2,095** | |

#### 7. Verification

✅ **All parameters are trainable**
✅ **No parameter sharing/tying**
✅ **GCN weight matrices are eliminated** (replaced by quantum circuits)
✅ **Feature reduction matrices are NOT tied**

**Code verification**:
- Feature reduction: `src/models/QCN_Layers/QGCNConv.py` line 43 (bias=False)
- Quantum circuit: `src/models/QNN_Node_Embedding.py` line 66 (weight_shapes)
- Bias: `src/models/QCN_Layers/QGCNConv.py` line 48
- Classifiers: `src/models/Quantum_Classifiers/MPS.py` line 53, `TTN.py` line 66

---

## Summary

### For input_dim = 250:
- **Number of qubits**: 8 (fixed)
- **Current config**: ~2,039 parameters
- **Maximum config**: ~2,095 parameters

### Key Points:
1. **L = number of QGCNConv layers** (1 or 2 in current implementation)
2. **Feature dimensions**: 250 → 8 → 1
3. **GCN weight matrices**: **ELIMINATED** (replaced by quantum circuits)
4. **Parameter tying**: **NO** (all parameters are independent)
5. **All parameters are trainable**

The "2K total parameter claim" (~2,000 parameters) is **verified** for the current configuration with input_dim = 250.

