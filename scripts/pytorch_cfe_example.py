#!/usr/bin/env python3
"""
PyTorch CFES Example Script
===========================

This script demonstrates how to implement Controlled Functional Expansion Systems (CFES)
using PyTorch with tensorboard logging and comparison to JAX/diffrax implementations.

Requirements:
- torch
- tensorboard
- matplotlib
- numpy
- diffrax (for comparison)

Usage:
    python scripts/pytorch_cfe_example.py
    # Or run sections individually in Jupyter or IDE with #%% support
"""

#%%
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import time
from typing import Tuple, Dict, Any, Optional
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#%%
# Configuration
class PyTorchConfig:
    # Data generation
    dataset_size = 1000
    sequence_length = 100
    input_dim = 3  # time + 2D coordinates
    output_dim = 1
    
    # Model
    hidden_dim = 64
    num_layers = 2
    dropout = 0.1
    
    # Training
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100
    weight_decay = 1e-5
    
    # TensorBoard
    log_dir = "logs/pytorch_cfe"
    save_freq = 10
    
    # Visualization
    save_plots = True
    plot_format = 'png'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = PyTorchConfig()

# Create log directory
os.makedirs(config.log_dir, exist_ok=True)

#%%
# PyTorch Neural CDE Model
class PyTorchFunc(nn.Module):
    """Vector field function for PyTorch Neural CDE."""
    
    def __init__(self, input_dim: int, hidden_dim: int, width: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.Tanh(),  # Important: prevent blowup like in JAX version
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, hidden_dim * input_dim),
        )
    
    def forward(self, t: torch.Tensor, y: torch.Tensor, params: Any = None) -> torch.Tensor:
        batch_size = y.shape[0]
        output = self.net(y)
        return output.view(batch_size, self.hidden_dim, self.input_dim)


class PyTorchNeuralCDE(nn.Module):
    """PyTorch implementation of Neural CDE."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, width: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initial mapping from input to hidden state
        self.initial = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, hidden_dim)
        )
        
        # Vector field function
        self.func = PyTorchFunc(input_dim, hidden_dim, width)
        
        # Output mapping
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # For binary classification
        )
    
    def forward(self, ts: torch.Tensor, x: torch.Tensor, return_path: bool = False) -> torch.Tensor:
        """
        Forward pass through the Neural CDE.
        
        Args:
            ts: Time points (batch_size, seq_len)
            x: Input paths (batch_size, seq_len, input_dim)
            return_path: If True, return the entire path; if False, return only final state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initial condition
        y0 = self.initial(x[:, 0, :])  # (batch_size, hidden_dim)
        
        # Use Euler method for solving CDE (simplified)
        if return_path:
            ys = [y0]
        
        y = y0
        for i in range(seq_len - 1):
            dt = ts[:, i+1] - ts[:, i]  # (batch_size,)
            
            # Evaluate vector field
            f = self.func(ts[:, i], y)  # (batch_size, hidden_dim, input_dim)
            
            # Simple Euler integration
            dx = x[:, i+1, :] - x[:, i, :]  # (batch_size, input_dim)
            dy = torch.einsum('bij,bj->bi', f, dx)  # (batch_size, hidden_dim)
            
            y = y + dy * dt.unsqueeze(-1)
            
            if return_path:
                ys.append(y)
        
        if return_path:
            return torch.stack(ys, dim=1)  # (batch_size, seq_len, hidden_dim)
        else:
            return self.output(y)  # (batch_size, output_dim)


#%%
# Data Generation (PyTorch version)
def generate_spiral_data_pytorch(config: PyTorchConfig, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate spiral data using PyTorch."""
    dataset_size = config.dataset_size
    seq_len = config.sequence_length
    input_dim = config.input_dim
    
    # Generate random phases
    theta = 2 * math.pi * torch.rand(dataset_size)
    
    # Initial conditions
    y0 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    
    # Time vector
    ts = torch.linspace(0, 4 * math.pi, seq_len).unsqueeze(0).repeat(dataset_size, 1)
    
    # Spiral dynamics matrix
    matrix = torch.tensor([[-0.3, 2.0], [-2.0, -0.3]])
    
    # Generate spirals
    ys = []
    for i in range(dataset_size):
        y_traj = []
        for j in range(seq_len):
            t = ts[i, j]
            # Matrix exponential (simplified with small dt approximation)
            exp_mt = torch.matrix_exp(t * matrix)
            y = exp_mt @ y0[i]
            y_traj.append(y)
        ys.append(torch.stack(y_traj))
    
    ys = torch.stack(ys)  # (dataset_size, seq_len, 2)
    
    # Add time as a channel
    ts_expanded = ts.unsqueeze(-1)
    data = torch.cat([ts_expanded, ys], dim=-1)  # (dataset_size, seq_len, 3)
    
    # Make half the spirals counter-clockwise
    data[:dataset_size//2, :, 1] *= -1
    
    # Labels: clockwise = 1, counter-clockwise = 0
    labels = torch.zeros(dataset_size)
    labels[:dataset_size//2] = 1.0
    
    return data, labels, ts


class SpiralDataset(torch.utils.data.Dataset):
    """PyTorch dataset for spiral data."""
    
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, ts: torch.Tensor):
        self.data = data
        self.labels = labels
        self.ts = ts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.ts[idx], self.data[idx], self.labels[idx]


#%%
# Training utilities
def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss."""
    return F.binary_cross_entropy(predictions.squeeze(), targets)


def train_epoch(model: PyTorchNeuralCDE, dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, device: str) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for ts, data, labels in dataloader:
        ts = ts.to(device)
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(ts, data)
        
        loss = compute_loss(predictions, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        accuracy = ((predictions.squeeze() > 0.5) == (labels == 1)).float().mean().item()
        total_accuracy += accuracy
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def evaluate_model(model: PyTorchNeuralCDE, dataloader: torch.utils.data.DataLoader, 
                   device: str) -> Tuple[float, float]:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for ts, data, labels in dataloader:
            ts = ts.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            predictions = model(ts, data)
            loss = compute_loss(predictions, labels)
            
            total_loss += loss.item()
            accuracy = ((predictions.squeeze() > 0.5) == (labels == 1)).float().mean().item()
            total_accuracy += accuracy
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy

#%%
# Training function with TensorBoard logging
def train_pytorch_cfe(config: PyTorchConfig) -> Tuple[PyTorchNeuralCDE, Dict]:
    """Train PyTorch Neural CDE with TensorBoard logging."""
    print("Starting PyTorch Neural CDE training...")
    
    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating spiral data...")
    data, labels, ts = generate_spiral_data_pytorch(config, device)
    
    # Split into train/val
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    
    train_data, val_data = torch.utils.data.random_split(
        SpiralDataset(data, labels, ts), [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config.batch_size, shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = PyTorchNeuralCDE(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim
    ).to(device)
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        # Log to console
        print(f"Epoch {epoch+1:3d}/{config.num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
            }, os.path.join(config.log_dir, 'best_model.pth'))
            print(f"  -> New best model saved with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(config.log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'model': model
    }
    
    return model, results

#%%
# Visualization functions
def plot_pytorch_training_history(results: Dict):
    """Plot training history for PyTorch model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(results['train_losses'], 'b-', linewidth=2, label='Train')
    ax1.plot(results['val_losses'], 'r-', linewidth=2, label='Validation')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(results['train_accuracies'], 'b-', linewidth=2, label='Train')
    ax2.plot(results['val_accuracies'], 'r-', linewidth=2, label='Validation')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.save_plots:
        plt.savefig(f'pytorch_training_history.{config.plot_format}', dpi=300, bbox_inches='tight')
        print(f"PyTorch training history plot saved as pytorch_training_history.{config.plot_format}")
    
    plt.show()


def visualize_spiral_predictions_pytorch(model: PyTorchNeuralCDE, config: PyTorchConfig, device: str = 'cpu'):
    """Visualize spiral predictions from PyTorch model."""
    model.eval()
    
    # Generate a few examples
    with torch.no_grad():
        data, labels, ts = generate_spiral_data_pytorch(PyTorchConfig(), device)
        
        # Take first 4 examples
        sample_data = data[:4].to(device)
        sample_ts = ts[:4].to(device)
        sample_labels = labels[:4]
        
        predictions = model(sample_ts, sample_data)
        
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Extract data
        t = sample_ts[i].cpu().numpy()
        x = sample_data[i, :, 1].cpu().numpy()  # x coordinate
        y = sample_data[i, :, 2].cpu().numpy()  # y coordinate
        
        # Plot spiral
        ax.plot(t, x, y, 'b-', linewidth=2, alpha=0.7)
        
        # Color by prediction confidence
        pred = predictions[i].item()
        color = plt.cm.viridis(pred)
        
        # Add prediction as colored points
        ax.scatter(t[::5], x[::5], y[::5], c=[color]*len(t[::5]), 
                  s=50, alpha=0.8, depthshade=False)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.set_title(f'Sample {i+1}\nTrue: {sample_labels[i]:.0f}, Pred: {pred:.3f}')
    
    plt.tight_layout()
    
    if config.save_plots:
        plt.savefig(f'pytorch_spiral_predictions.{config.plot_format}', dpi=300, bbox_inches='tight')
        print(f"PyTorch spiral predictions plot saved as pytorch_spiral_predictions.{config.plot_format}")
    
    plt.show()


def compare_frameworks():
    """Compare PyTorch and JAX implementations (placeholder)."""
    print("Framework comparison would be implemented here.")
    print("- Training speed comparison")
    print("- Memory usage comparison") 
    print("- Accuracy comparison")
    print("- Code complexity comparison")

#%%
# Main execution
def main():
    """Main training and evaluation pipeline."""
    print("PyTorch Neural CDE Example")
    print("==========================")
    print(f"Configuration: {config.__dict__}")
    
    # Train model
    model, results = train_pytorch_cfe(config)
    
    # Plot results
    plot_pytorch_training_history(results)
    visualize_spiral_predictions_pytorch(model, config)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {max(results['val_accuracies']):.3f}")
    print(f"Final validation accuracy: {results['val_accuracies'][-1]:.3f}")
    
    # Compare frameworks
    compare_frameworks()
    
    return model, results

#%%
# Run the example
if __name__ == "__main__":
    model, results = main()
    
    print("\n" + "="*50)
    print("PyTorch CFES Example completed successfully!")
    print(f"TensorBoard logs saved to: {config.log_dir}")
    print("To view TensorBoard: tensorboard --logdir logs/pytorch_cfe")
    print("="*50)