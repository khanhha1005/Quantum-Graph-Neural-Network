"""
Calculate the number of parameters for the Quantum GCN model.
Provides detailed parameter counting methodology.
"""
import numpy as np

def calculate_params(n_qubits, input_dims, q_depths, output_dims, classifier_type="TTN"):
    """
    Calculate total parameters for QGCN model.
    
    Args:
        n_qubits: Number of qubits (4 or 8)
        input_dims: Input feature dimensionality
        q_depths: List of quantum layer depths for each QGCNConv layer
        output_dims: Output dimensions (1 for binary classification)
        classifier_type: "MPS", "TTN", or "Linear"
    
    Returns:
        Dictionary with parameter breakdown
    """
    total_params = 0
    breakdown = {
        'qgcn_layers': [],
        'classifier': {},
        'total': 0
    }
    
    # QGCNConv layers
    for i, q_depth in enumerate(q_depths):
        layer_input_dims = input_dims if i == 0 else n_qubits
        
        # Feature reduction layer (if needed)
        if layer_input_dims != n_qubits:
            feature_reduction_params = layer_input_dims * n_qubits
            # Note: bias=False in QGCNConv (line 43)
        else:
            feature_reduction_params = 0
        
        # Quantum circuit parameters
        # From quantum_net: weight_shapes = {"q_weights": (n_layers, n_qubits, 2)}
        # Each layer has RY and RZ rotations per qubit = n_layers × n_qubits × 2
        quantum_params = q_depth * n_qubits * 2
        
        # Bias parameters (n_qubits learnable bias values)
        bias_params = n_qubits
        
        layer_params = feature_reduction_params + quantum_params + bias_params
        total_params += layer_params
        
        breakdown['qgcn_layers'].append({
            'layer': i + 1,
            'input_dims': layer_input_dims,
            'feature_reduction': feature_reduction_params,
            'quantum_circuit': quantum_params,
            'bias': bias_params,
            'total': layer_params
        })
    
    # Classifier parameters
    if classifier_type == "MPS":
        # MPS: 2*n_qubits - 1 parameters
        classifier_params = 2 * n_qubits - 1
    elif classifier_type == "TTN":
        # TTN: 2 * (n_qubits - 1) + 1 parameters
        n_layers_ttn = int(np.log2(n_qubits))
        total_pairs = 0
        for layer in range(n_layers_ttn):
            n_gates = n_qubits // (2 ** (layer + 1))
            total_pairs += n_gates
        classifier_params = 2 * total_pairs + 1
    else:  # Linear
        # Linear: n_qubits * output_dims
        classifier_params = n_qubits * output_dims
    
    total_params += classifier_params
    breakdown['classifier'] = {
        'type': classifier_type,
        'params': classifier_params
    }
    breakdown['total'] = total_params
    
    return breakdown, total_params


def print_detailed_breakdown(n_qubits, input_dims, q_depths, output_dims, classifier_type="TTN"):
    """Print detailed parameter breakdown."""
    breakdown, total = calculate_params(n_qubits, input_dims, q_depths, output_dims, classifier_type)
    
    print("="*80)
    print("QUANTUM GCN MODEL PARAMETER COUNTING METHODOLOGY")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input feature dimensions: {input_dims}")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Quantum layer depths: {q_depths} (L = {len(q_depths)} QGCNConv layers)")
    print(f"  Output dimensions: {output_dims}")
    print(f"  Classifier type: {classifier_type}")
    
    print(f"\n{'─'*80}")
    print("PARAMETER BREAKDOWN:")
    print(f"{'─'*80}")
    
    # QGCNConv layers
    print(f"\n1. QGCNConv Layers (L = {len(q_depths)} layers):")
    print(f"   {'─'*76}")
    for layer_info in breakdown['qgcn_layers']:
        print(f"\n   Layer {layer_info['layer']}:")
        if layer_info['feature_reduction'] > 0:
            print(f"     - Feature reduction matrix: {layer_info['input_dims']} × {n_qubits} = {layer_info['feature_reduction']:,} params")
            print(f"       (Linear layer: input_dims → n_qubits, NO bias)")
        else:
            print(f"     - Feature reduction: Not needed (input_dims == n_qubits)")
        print(f"     - Quantum circuit weights: {q_depths[layer_info['layer']-1]} × {n_qubits} × 2 = {layer_info['quantum_circuit']:,} params")
        print(f"       (Variational layers: RY + RZ rotations per qubit per layer)")
        print(f"     - Bias vector: {n_qubits} params")
        print(f"     - Layer {layer_info['layer']} total: {layer_info['total']:,} params")
    
    qgcn_total = sum(l['total'] for l in breakdown['qgcn_layers'])
    print(f"\n   QGCNConv layers total: {qgcn_total:,} params")
    
    # Classifier
    print(f"\n2. Classifier ({classifier_type}):")
    print(f"   {'─'*76}")
    if classifier_type == "MPS":
        print(f"   - MPS quantum classifier: 2 × {n_qubits} - 1 = {breakdown['classifier']['params']:,} params")
        print(f"     (Sequential qubit entanglement with trainable RY rotations)")
    elif classifier_type == "TTN":
        n_layers_ttn = int(np.log2(n_qubits))
        total_pairs = 0
        for layer in range(n_layers_ttn):
            n_gates = n_qubits // (2 ** (layer + 1))
            total_pairs += n_gates
        print(f"   - TTN quantum classifier: 2 × {total_pairs} + 1 = {breakdown['classifier']['params']:,} params")
        print(f"     (Tree tensor network with {n_layers_ttn} layers, {total_pairs} qubit pairs)")
    else:
        print(f"   - Linear classifier: {n_qubits} × {output_dims} = {breakdown['classifier']['params']:,} params")
    
    print(f"\n{'─'*80}")
    print(f"TOTAL PARAMETERS: {total:,}")
    print(f"{'─'*80}")
    
    return total


# Answer the user's questions
print("\n" + "="*80)
print("QUESTION 1: Parameters for input_dim = 250")
print("="*80)

# Current configuration (q_depths=[1])
n_qubits = 8  # Fixed at 8 for input_dims > 4
input_dims_250 = 250
q_depths_current = [1]
output_dims = 1

print("\nCurrent Configuration (q_depths=[1]):")
print("─"*80)
for classifier in ["MPS", "TTN", "Linear"]:
    total = print_detailed_breakdown(n_qubits, input_dims_250, q_depths_current, output_dims, classifier)
    print()

# Maximum configuration (q_depths=[2, 2])
print("\n" + "="*80)
print("Maximum Configuration (q_depths=[2, 2]):")
print("="*80)
q_depths_max = [2, 2]

for classifier in ["MPS", "TTN", "Linear"]:
    total = print_detailed_breakdown(n_qubits, input_dims_250, q_depths_max, output_dims, classifier)
    print()

# Address the methodology question
print("\n" + "="*80)
print("QUESTION 2: Parameter Counting Methodology Specification")
print("="*80)
print("""
ADDRESSING THE CONCERN: "The parameter counting methodology is not fully 
specified (e.g., number of layers L, feature dimensionalities, whether GCN 
weight matrices are eliminated or tied), making it difficult to verify the 
2K total parameter claim."

CLARIFICATIONS:

1. NUMBER OF LAYERS (L):
   - L = len(q_depths) = number of QGCNConv layers
   - Current config: L = 1 (q_depths=[1])
   - Maximum config: L = 2 (q_depths=[2, 2])
   - Each QGCNConv layer contains:
     * Feature reduction matrix (if input_dims ≠ n_qubits)
     * Quantum variational circuit
     * Bias vector

2. FEATURE DIMENSIONALITIES:
   - Input: input_dims (e.g., 250 from node features)
   - After first layer: reduced to n_qubits (8)
   - Subsequent layers: n_qubits → n_qubits (no reduction needed)
   - Output: n_qubits (8) → output_dims (1) via classifier

3. GCN WEIGHT MATRICES:
   - NO traditional GCN weight matrices (W matrices are ELIMINATED)
   - Instead: Quantum circuits replace the linear transformation
   - Message passing uses normalized adjacency (no learnable weights)
   - Only bias vectors are learnable in message passing step

4. PARAMETER TYING:
   - Parameters are NOT tied across layers
   - Each QGCNConv layer has independent parameters
   - Each layer has its own:
     * Feature reduction matrix (if needed)
     * Quantum circuit weights
     * Bias vector

5. PARAMETER COUNTING FORMULA:
   For L QGCNConv layers:
   Total = Σ[layer=1 to L] {
       (input_dims[layer] × n_qubits) if input_dims[layer] ≠ n_qubits else 0
       + (q_depth[layer] × n_qubits × 2)  // Quantum circuit
       + n_qubits  // Bias
   }
   + classifier_params

6. VERIFICATION:
   - All parameters are trainable
   - No parameter sharing/tying
   - GCN weight matrices are eliminated (replaced by quantum circuits)
   - Feature reduction matrices are NOT tied
""")

print("\n" + "="*80)
print("SUMMARY FOR input_dim = 250:")
print("="*80)
print(f"Number of qubits: {n_qubits} (fixed)")
print(f"\nCurrent config (q_depths=[1], L=1):")
for classifier in ["MPS", "TTN", "Linear"]:
    _, total = calculate_params(n_qubits, input_dims_250, q_depths_current, output_dims, classifier)
    print(f"  - {classifier}: {total:,} params")

print(f"\nMaximum config (q_depths=[2, 2], L=2):")
for classifier in ["MPS", "TTN", "Linear"]:
    _, total = calculate_params(n_qubits, input_dims_250, q_depths_max, output_dims, classifier)
    print(f"  - {classifier}: {total:,} params")
