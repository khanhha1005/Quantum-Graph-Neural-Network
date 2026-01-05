import pennylane as qml
from pennylane.exceptions import DeviceError
import numpy as np
import torch


def TTN(n_qubits, device_name=None):
    """
    Tree Tensor Network quantum classifier optimized with lightning and adjoint differentiation.
    Precomputes qubit pairs outside QNode to reduce Python overhead.
    
    Args:
        n_qubits: Number of qubits (must be power of 2)
        device_name: Quantum device name (default: "lightning.gpu" with fallback to "lightning.qubit")
    """
    # Try GPU first, fallback to CPU if not available (e.g., on Windows)
    if device_name is None:
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits, shots=None)
            device_name = "lightning.gpu"
            print("Using lightning.gpu device")
        except (DeviceError, ImportError, Exception):
            # Fallback to CPU device
            dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
            device_name = "lightning.qubit"
            print("GPU device not available, falling back to lightning.qubit (CPU)")
    else:
        # Use specified device
        dev = qml.device(device_name, wires=n_qubits, shots=None)
    
    # Precompute qubit pairs outside QNode to reduce Python overhead
    n_layers = int(np.log2(n_qubits))
    pairs = []
    i = 0
    for layer in range(n_layers):
        n_gates = n_qubits // (2 ** (layer + 1))
        for j in range(n_gates):
            step = n_qubits // (2 ** (n_layers - layer - 1))
            qubit0 = j * step + (2 ** layer) - 1
            qubit1 = j * step + (2 ** (layer + 1)) - 1
            pairs.append((qubit0, qubit1, i))
            i += 2
    
    n_params = 2 * len(pairs) + 1

    @qml.qnode(dev, interface="torch", diff_method="adjoint", cache=True)
    def quantum_circuit(inputs, q_weights_flat):
        """
        The variational quantum classifier.
        """

        # Embed features in the quantum node
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Use precomputed pairs instead of recalculating in each forward pass
        for q0, q1, idx in pairs:
            qml.RY(q_weights_flat[idx], wires=q0)
            qml.RY(q_weights_flat[idx + 1], wires=q1)
            qml.CZ(wires=[q0, q1])

        qml.RY(q_weights_flat[-1], wires=n_qubits-1)

        # Expectation values in the Z basis
        return [qml.expval(qml.PauliZ(n_qubits - 1))]

    return qml.qnn.TorchLayer(quantum_circuit, {"q_weights_flat": n_params}), quantum_circuit
