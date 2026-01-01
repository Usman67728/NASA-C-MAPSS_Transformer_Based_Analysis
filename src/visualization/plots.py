"""
Visualization utilities for RUL prediction.

Provides functions for creating publication-quality plots:
- Attention heatmaps
- Prediction vs True RUL plots
- Error distribution
- Training curves
- Feature/Sensor importance
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def plot_attention_heatmap(
    attention: np.ndarray,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention: Attention matrix of shape [seq_len, seq_len].
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
        cmap: Colormap to use.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attention, cmap=cmap, aspect='auto')
    
    ax.set_xlabel("Key Position (Time Step)", fontsize=12)
    ax.set_ylabel("Query Position (Time Step)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_multi_head_attention(
    attention: np.ndarray,
    n_heads: int = 8,
    layer_name: str = "Layer 1",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Plot attention from multiple heads in a grid.
    
    Args:
        attention: Attention tensor of shape [n_heads, seq_len, seq_len].
        n_heads: Number of attention heads.
        layer_name: Layer name for title.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        im = ax.imshow(attention[head_idx], cmap='Blues', aspect='auto')
        ax.set_title(f"Head {head_idx + 1}", fontsize=10)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
    
    # Remove empty subplots
    for idx in range(n_heads, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(f"Multi-Head Attention - {layer_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_predictions_vs_true(
    predictions: np.ndarray,
    true_rul: np.ndarray,
    max_rul: int = 125,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot predicted vs true RUL scatter plot.
    
    Args:
        predictions: Predicted RUL values.
        true_rul: True RUL values.
        max_rul: Maximum RUL for axis limits.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    scatter = ax.scatter(
        true_rul, predictions,
        alpha=0.6, c='steelblue', edgecolors='navy', linewidth=0.5
    )
    
    # Perfect prediction line
    ax.plot([0, max_rul], [0, max_rul], 'r--', linewidth=2, label='Ideal Prediction')
    
    # Error bands
    ax.fill_between(
        [0, max_rul], [10, max_rul + 10], [-10, max_rul - 10],
        alpha=0.1, color='green', label='±10 cycles'
    )
    
    ax.set_xlabel("True RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
    ax.set_title("Predicted vs True Remaining Useful Life", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, max_rul + 10)
    ax.set_ylim(0, max_rul + 30)
    ax.grid(True, alpha=0.3)
    
    # Add RMSE annotation
    rmse = np.sqrt(np.mean((predictions - true_rul) ** 2))
    ax.text(
        0.95, 0.05, f"RMSE: {rmse:.2f}",
        transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_error_distribution(
    predictions: np.ndarray,
    true_rul: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot error distribution histogram.
    
    Args:
        predictions: Predicted RUL values.
        true_rul: True RUL values.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    errors = predictions - true_rul
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    n, bins, patches = ax.hist(
        errors, bins=30, color='steelblue', edgecolor='navy',
        alpha=0.7, density=True
    )
    
    # Color overestimation differently
    for i, patch in enumerate(patches):
        if bins[i] >= 0:
            patch.set_facecolor('coral')
    
    # Add KDE
    from scipy import stats
    kde_x = np.linspace(errors.min(), errors.max(), 100)
    kde = stats.gaussian_kde(errors)
    ax.plot(kde_x, kde(kde_x), 'k-', linewidth=2, label='KDE')
    
    # Zero line
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    ax.set_xlabel("Prediction Error (Predicted - True)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of RUL Prediction Errors", fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add statistics
    stats_text = f"Mean: {errors.mean():.2f}\nStd: {errors.std():.2f}\nOver-rate: {(errors > 0).mean():.1%}"
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RMSE curve
    axes[1].plot(epochs, history['val_rmse'], 'g-', linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (cycles)")
    axes[1].set_title("Validation RMSE")
    axes[1].grid(True, alpha=0.3)
    
    # Overestimation rate
    over_rates = [r * 100 for r in history['val_over_rate']]
    axes[2].plot(epochs, over_rates, 'm-', linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Overestimation Rate (%)")
    axes[2].set_title("Validation Overestimation Rate")
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Training Progress", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_sensor_importance(
    importance_scores: Dict[str, float],
    config: Optional[Config] = None,
    top_k: int = 15,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot sensor importance bar chart.
    
    Args:
        importance_scores: Dictionary of feature -> importance.
        config: Configuration object for subsystem mapping.
        top_k: Number of top features to show.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    config = config or Config()
    
    # Sort by importance
    sorted_items = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_k]
    
    features = [item[0] for item in top_items]
    scores = [item[1] for item in top_items]
    
    # Get subsystem colors
    subsystem_colors = {
        'Fan': '#FF6B6B',
        'Low-Pressure Compressor (LPC)': '#4ECDC4',
        'High-Pressure Compressor (HPC)': '#45B7D1',
        'Combustion Chamber': '#F7DC6F',
        'High-Pressure Turbine (HPT)': '#BB8FCE',
        'Overall Inlet': '#85C1E9',
        'Overall Engine System': '#82E0AA',
        'HPC and HPT': '#F8B500',
        'default': '#95A5A6'
    }
    
    colors = []
    for feat in features:
        subsystems = config.SENSOR_TO_SUBSYSTEM.get(feat, ['default'])
        subsystem = subsystems[-1] if subsystems else 'default'
        colors.append(subsystem_colors.get(subsystem, subsystem_colors['default']))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(range(len(features)), scores, color=colors, edgecolor='navy', linewidth=0.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Sensor Importance for RUL Prediction", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_temporal_attention(
    temporal_importance: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot attention importance over time steps.
    
    Args:
        temporal_importance: Importance scores per time step.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time_steps = range(len(temporal_importance))
    
    ax.bar(time_steps, temporal_importance, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.plot(time_steps, temporal_importance, 'r-o', markersize=4, linewidth=1.5)
    
    ax.set_xlabel("Time Step (Cycle)", fontsize=12)
    ax.set_ylabel("Attention Importance", fontsize=12)
    ax.set_title("Temporal Attention Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate recent cycles importance
    recent_importance = temporal_importance[-5:].sum()
    ax.text(
        0.95, 0.95, f"Last 5 cycles: {recent_importance:.1%}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_all_visualizations(
    model,
    X_test: np.ndarray,
    predictions: np.ndarray,
    true_rul: np.ndarray,
    history: Dict[str, List[float]],
    config: Optional[Config] = None,
    output_dir: Optional[str] = None
):
    """
    Create all visualizations and save to output directory.
    
    Args:
        model: Trained model.
        X_test: Test sequences.
        predictions: Model predictions.
        true_rul: True RUL values.
        history: Training history.
        config: Configuration object.
        output_dir: Output directory for plots.
    """
    config = config or Config()
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Training curves
    if history:
        plot_training_curves(
            history,
            save_path=os.path.join(output_dir, 'training_curves.png')
        )
    
    # Predictions vs True
    plot_predictions_vs_true(
        predictions, true_rul,
        save_path=os.path.join(output_dir, 'predictions_vs_true.png')
    )
    
    # Error distribution
    plot_error_distribution(
        predictions, true_rul,
        save_path=os.path.join(output_dir, 'error_distribution.png')
    )
    
    # Attention heatmap (first test sample)
    model.eval()
    with torch.no_grad():
        X_sample = torch.tensor(X_test[0:1], dtype=torch.float32).to(config.DEVICE)
        _, attention_weights = model(X_sample, return_attention=True)
        
        if attention_weights:
            # Aggregate attention
            agg_attn = attention_weights[-1][0].mean(dim=0).cpu().numpy()
            plot_attention_heatmap(
                agg_attn,
                title="Attention Weights (Last Layer, Averaged)",
                save_path=os.path.join(output_dir, 'attention_heatmap.png')
            )
            
            # Multi-head attention
            plot_multi_head_attention(
                attention_weights[-1][0].cpu().numpy(),
                n_heads=config.N_HEADS,
                layer_name="Last Encoder Layer",
                save_path=os.path.join(output_dir, 'multi_head_attention.png')
            )
    
    print(f"Visualizations saved to: {output_dir}")
    plt.close('all')
