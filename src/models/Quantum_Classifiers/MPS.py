import pennylane as qml
from pennylane.exceptions import DeviceError
import torch


def MPS(n_qubits, meas_qubits, device_name=None):
    """
    Matrix Product State quantum classifier optimized with lightning and adjoint differentiation.
    
    Args:
        n_qubits: Number of qubits
        meas_qubits: List of qubit indices to measure
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

    @qml.qnode(dev, interface="torch", diff_method="adjoint", cache=True)
    def quantum_circuit(inputs, q_weights_flat):
        """
        The variational quantum classifier.
        """

        # Reshape weights
        q_weights = q_weights_flat[:-1].reshape(n_qubits-1, 2)

        # Embed features in the quantum node
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Sequence of trainable variational layers
        for k in range(n_qubits-1):
            qml.RY(q_weights[k][0], wires=k)
            qml.RY(q_weights[k][1], wires=k+1)
            qml.CZ(wires=[k, k+1])

        qml.RY(q_weights_flat[-1], wires=n_qubits-1)

        # Expectation values in the Z basis
        return [qml.expval(qml.PauliZ(qbit_i)) for qbit_i in meas_qubits]

    return qml.qnn.TorchLayer(quantum_circuit, {"q_weights_flat": (2*n_qubits - 1)}), quantum_circuit
