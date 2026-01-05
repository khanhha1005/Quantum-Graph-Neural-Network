import os
import sys

# Fix OpenMP conflict on Windows (must be set before importing torch/numpy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.Quantum_GCN import QGCN
from dataloader.load_vulnerability_data import get_dataloaders, load_vulnerability_data


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics."""
    # Convert to numpy - handle lists, tensors, and numpy arrays
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    elif isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    else:
        y_true = np.array(y_true)
    
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    elif isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred = np.array(y_pred)
    
    if y_proba is not None:
        if isinstance(y_proba, list):
            y_proba = np.array(y_proba)
        elif isinstance(y_proba, torch.Tensor):
            y_proba = y_proba.cpu().numpy()
        else:
            y_proba = np.array(y_proba)
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch=None, writer=None):
    """Train for one epoch."""
    import time
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = None
    
    # Add progress bar for training batches
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Show timing for first batch
        if batch_idx == 0:
            start_time = time.time()
        
        # Forward pass
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # For binary classification, squeeze to [batch_size] for BCEWithLogitsLoss
        if out.dim() > 1 and out.size(1) == 1:
            out = out.squeeze(1)
        
        loss = criterion(out, batch.y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store predictions and labels
        # For binary classification, use sigmoid threshold
        if out.dim() == 1:
            preds = (torch.sigmoid(out) > 0.5).long()
        else:
            preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu())
        all_labels.extend(batch.y.cpu())
        
        total_loss += loss.item()
        
        # Update progress bar with timing info
        if batch_idx == 0:
            elapsed = time.time() - start_time
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'time': f'{elapsed:.1f}s',
                'NOTE': 'Quantum circuits are slow on CPU'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics


def validate_epoch(model, val_loader, criterion, device, epoch=None):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Add progress bar for validation batches
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # For binary classification, squeeze to [batch_size] for BCEWithLogitsLoss
            if out.dim() > 1 and out.size(1) == 1:
                out = out.squeeze(1)
            
            loss = criterion(out, batch.y.float())
            
            # Store predictions and labels
            # For binary classification, use sigmoid threshold
            if out.dim() == 1:
                preds = (torch.sigmoid(out) > 0.5).long()
            else:
                preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu())
            all_labels.extend(batch.y.cpu())
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics


def train_model(vulnerability_type, classifier_type, train_loader, val_loader, config, log_dir=None):
    """Train a quantum model for a specific vulnerability type."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training {classifier_type} model for {vulnerability_type}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Setup TensorBoard writer
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    
    # Get input dimensions from first batch
    sample_batch = next(iter(train_loader))
    input_dims = sample_batch.x.size(1)
    output_dims = 1  # Binary classification
    
    # Model configuration
    q_depths = config.get('q_depths', [2, 2])  # Quantum layer depths
    
    # Handle 'Linear' classifier (pass None or empty string for Linear)
    classifier_param = classifier_type if classifier_type in ['MPS', 'TTN'] else None
    
    # Create model
    model = QGCN(
        input_dims=input_dims,
        q_depths=q_depths,
        output_dims=output_dims,
        classifier=classifier_param,
        readout=False
    ).to(device)
    
    print(f"Model created with input_dims={input_dims}, output_dims={output_dims}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    print(f"Quantum layer depths: {q_depths}")
    print(f"Batch size: {config.get('batch_size', 32)}")
    print(f"\n⚠️  NOTE: Quantum circuit execution is computationally intensive on CPU.")
    print(f"   Each node executes a quantum circuit with {sum(q_depths)} total layers.")
    print(f"   Optimizations applied:")
    print(f"     - Batch size: {config.get('batch_size', 32)}")
    print(f"     - Quantum depths: {q_depths}")
    print(f"     - Validate every: {config.get('validate_every', 1)} epochs")
    print(f"     - Max nodes per batch: {config.get('max_nodes_per_batch', 'unlimited')}")
    print(f"   Expected time per batch: 5-60+ seconds depending on graph size.\n")
    
    # Log model info to TensorBoard
    if writer:
        writer.add_text('Model/Classifier', classifier_type, 0)
        writer.add_text('Model/Vulnerability_Type', vulnerability_type, 0)
        writer.add_scalar('Model/Num_Parameters', num_params, 0)
        writer.add_scalar('Model/Input_Dims', input_dims, 0)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 15),
        min_delta=config.get('min_delta', 0.0),
        restore_best_weights=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # Training loop
    max_epochs = config.get('max_epochs', 100)
    validate_every = config.get('validate_every', 1)  # Validate every N epochs
    best_val_f1 = 0.0
    best_model_state = None  # Store best model state
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(1, max_epochs + 1), desc=f"{vulnerability_type}-{classifier_type}")
    
    for epoch in epoch_pbar:
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, writer=writer
        )
        
        # Validate only every N epochs to save time
        if epoch % validate_every == 0 or epoch == 1 or epoch == max_epochs:
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch=epoch
            )
        else:
            # Use last validation metrics if skipping
            val_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else 0
            val_metrics = {
                'accuracy': history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0,
                'precision': history.get('val_precision', [0])[-1] if history.get('val_precision') else 0,
                'recall': history.get('val_recall', [0])[-1] if history.get('val_recall') else 0,
                'f1': history.get('val_f1', [0])[-1] if history.get('val_f1') else 0,
            }
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1'])
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Log to TensorBoard (only when validating or every 5 epochs to reduce overhead)
        if writer and (epoch % validate_every == 0 or epoch % 5 == 0 or epoch == 1 or epoch == max_epochs):
            # Training metrics
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Precision/Train', train_metrics['precision'], epoch)
            writer.add_scalar('Recall/Train', train_metrics['recall'], epoch)
            writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
            
            # Validation metrics (only when actually validating)
            if epoch % validate_every == 0 or epoch == 1 or epoch == max_epochs:
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
                writer.add_scalar('Precision/Validation', val_metrics['precision'], epoch)
                writer.add_scalar('Recall/Validation', val_metrics['recall'], epoch)
                writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_f1': f'{val_metrics["f1"]:.4f}'
        })
        
        # Print detailed progress (every N epochs or first/last epoch)
        print_interval = max(validate_every, 5)
        if epoch == 1 or epoch % print_interval == 0 or epoch == max_epochs:
            val_status = "✓" if (epoch % validate_every == 0 or epoch == 1 or epoch == max_epochs) else "(skipped)"
            print(f"\nEpoch {epoch}/{max_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"  Val {val_status} - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
        
        # Track best validation F1 and save best model state
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()  # Save best model weights
            if writer:
                writer.add_scalar('Best/Validation_F1', best_val_f1, epoch)
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            if writer:
                writer.add_text('Training/Status', f'Early stopped at epoch {epoch}', epoch)
            break
    
    epoch_pbar.close()
    
    # Restore best model weights if early stopping was used or if we have a best state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model weights restored (Best validation F1: {best_val_f1:.4f})")
    else:
        print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    return model, history, best_model_state


def main():
    """Main training function."""
    
    # Optimize PyTorch for CPU
    torch.set_num_threads(torch.get_num_threads())
    torch.set_num_interop_threads(torch.get_num_threads())
    
    # Configuration - Optimized for speed
    config = {
        'batch_size': 64,  # Small batch size for quantum models (quantum circuits are slow)
        'learning_rate': 0.001,
        'max_epochs': 100,
        'patience': 10,
        'min_delta': 0.0001,
        'q_depths': [1],  # Single quantum layer for speed (was [1,1] or [2,2])
        'validate_every': 3,  # Validate every N epochs to save time
        'max_nodes_per_batch': 5000,  # Limit graph size to avoid huge batches
    }
    
    # Vulnerability types
    vulnerability_types = [ 'timestamp','reentrancy','integeroverflow']
    
    # Classifier types to train
    classifier_types = ['MPS', 'TTN', 'Linear']
    
    # Base directory for data
    base_data_dir = Path('train_data')
    
    # Create results directory
    results_dir = Path('training_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Train all combinations
    all_results = {}
    
    for vuln_type in vulnerability_types:
        print(f"\n{'#'*60}")
        print(f"Processing vulnerability type: {vuln_type}")
        print(f"{'#'*60}")
        
        train_path = base_data_dir / vuln_type / 'train.json'
        valid_path = base_data_dir / vuln_type / 'valid.json'
        
        if not train_path.exists() or not valid_path.exists():
            print(f"Warning: Data files not found for {vuln_type}. Skipping...")
            continue
        
        # Load data with progress bar and filtering
        print(f"Loading data for {vuln_type}...")
        with tqdm(total=2, desc="Loading datasets", leave=False) as pbar:
            # Load raw data
            train_data_raw = load_vulnerability_data(str(train_path))
            val_data_raw = load_vulnerability_data(str(valid_path))
            
            # Filter large graphs if max_nodes_per_batch is set
            if 'max_nodes_per_batch' in config:
                max_nodes = config['max_nodes_per_batch']
                train_data_filtered = [d for d in train_data_raw if d.x.size(0) <= max_nodes]
                val_data_filtered = [d for d in val_data_raw if d.x.size(0) <= max_nodes]
                print(f"Filtered: {len(train_data_filtered)}/{len(train_data_raw)} train samples "
                      f"({len(val_data_filtered)}/{len(val_data_raw)} val) with ≤{max_nodes} nodes")
                train_data_raw = train_data_filtered
                val_data_raw = val_data_filtered
            
            pbar.update(1)
            pbar.set_description(f"Loaded train: {len(train_data_raw)} samples")
            
            # Create optimized dataloaders (num_workers=0 for quantum circuits to avoid overhead)
            train_loader = DataLoader(train_data_raw, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_data_raw, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
            
            pbar.update(1)
            pbar.set_description(f"Loaded val: {len(val_data_raw)} samples")
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        vuln_results = {}
        
        for classifier_type in classifier_types:
            try:
                # Create model directory and TensorBoard log directory
                model_dir = results_dir / f"{vuln_type}_{classifier_type}_{timestamp}"
                model_dir.mkdir(exist_ok=True)
                log_dir = model_dir / 'logs'
                
                # Train model
                model, history, best_model_state = train_model(
                    vuln_type,
                    classifier_type,
                    train_loader,
                    val_loader,
                    config,
                    log_dir=str(log_dir)
                )
                
                # Save final model
                model_path = model_dir / 'model.pt'
                torch.save(model.state_dict(), model_path)
                
                # Save best model weights
                if best_model_state is not None:
                    best_model_path = model_dir / 'best_model.pt'
                    torch.save(best_model_state, best_model_path)
                    print(f"Best model weights saved to {best_model_path}")
                
                # Save history
                history_path = model_dir / 'history.json'
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                
                # Save config
                config_path = model_dir / 'config.json'
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"Model saved to {model_dir}")
                
                vuln_results[classifier_type] = {
                    'history': history,
                    'model_path': str(model_path),
                    'final_train_metrics': {
                        'loss': history['train_loss'][-1],
                        'accuracy': history['train_accuracy'][-1],
                        'precision': history['train_precision'][-1],
                        'recall': history['train_recall'][-1],
                        'f1': history['train_f1'][-1]
                    },
                    'final_val_metrics': {
                        'loss': history['val_loss'][-1],
                        'accuracy': history['val_accuracy'][-1],
                        'precision': history['val_precision'][-1],
                        'recall': history['val_recall'][-1],
                        'f1': history['val_f1'][-1]
                    },
                    'best_val_f1': max(history['val_f1'])
                }
                
            except Exception as e:
                print(f"Error training {classifier_type} for {vuln_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results[vuln_type] = vuln_results
    
    # Save summary
    summary_path = results_dir / f'summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for vuln_type, results in all_results.items():
        print(f"\n{vuln_type}:")
        for classifier_type, result in results.items():
            val_metrics = result['final_val_metrics']
            print(f"  {classifier_type}:")
            print(f"    Best Val F1: {result['best_val_f1']:.4f}")
            print(f"    Final Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"Prec: {val_metrics['precision']:.4f}, "
                  f"Rec: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
    
    print(f"\nAll results saved to {summary_path}")
    print(f"\nTensorBoard logs saved to: {results_dir}/[model_name]/logs/")
    print(f"To view TensorBoard, run: tensorboard --logdir={results_dir}")
    print(f"Training completed!")


if __name__ == '__main__':
    main()

