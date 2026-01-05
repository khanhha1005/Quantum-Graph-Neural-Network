import pennylane as qml
from pennylane.exceptions import DeviceError
import torch
import numpy as np


def quantum_net(n_qubits, n_layers, device_name=None):
    """
    Quantum variational circuit for node embedding.
    Creates a PennyLane TorchLayer that processes node features through a quantum circuit.
    
    Args:
        n_qubits: Number of qubits (should match input feature dimension)
        n_layers: Number of variational layers
        device_name: Quantum device name (default: "lightning.gpu" with fallback to "lightning.qubit")
    
    Returns:
        A PyTorch layer (qml.qnn.TorchLayer) that can process node embeddings
    """
    # Try GPU first, fallback to CPU if not available (e.g., on Windows)
    if device_name is None:
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits, shots=None)
            device_name = "lightning.gpu"
        except (DeviceError, ImportError, Exception):
            # Fallback to CPU device
            dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
            device_name = "lightning.qubit"
    else:
        # Use specified device
        dev = qml.device(device_name, wires=n_qubits, shots=None)
    
    @qml.qnode(dev, interface="torch", diff_method="adjoint", cache=True)
    def quantum_circuit(inputs, q_weights):
        """
        Quantum variational circuit for node embedding.
        
        Args:
            inputs: Node features [batch_size, n_qubits]
            q_weights: Trainable quantum weights [n_layers, n_qubits, 2]
        
        Returns:
            Expectation values for each qubit [batch_size, n_qubits]
        """
        # Embed features using angle embedding
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        
        # Apply variational layers
        for layer in range(n_layers):
            # Apply rotations to each qubit
            for qubit in range(n_qubits):
                qml.RY(q_weights[layer, qubit, 0], wires=qubit)
                qml.RZ(q_weights[layer, qubit, 1], wires=qubit)
            
            # Entangling layer - ring topology
            for qubit in range(n_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
            # Connect last qubit to first for ring topology
            if n_qubits > 2:
                qml.CZ(wires=[n_qubits - 1, 0])
        
        # Measure expectation values in Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Create weight shape: [n_layers, n_qubits, 2] (RY and RZ rotations per qubit per layer)
    weight_shapes = {"q_weights": (n_layers, n_qubits, 2)}
    
    # Return a TorchLayer that can be used as a PyTorch module
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

