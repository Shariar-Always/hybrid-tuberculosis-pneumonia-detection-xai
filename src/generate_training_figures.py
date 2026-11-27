"""
Publication-Grade Figure Generation Script: Training Dynamics & Convergence Curves
==================================================================================
Directly maps demo assets 'image copy.png' and 'image copy 10.png' to the
DenseNet-121 + CBAM Hybrid model trained on the 22,356 clinical CXR dataset.

Outputs:
  - reports/figures/training/fig3_1_densenet121_loss_convergence.png (.pdf)
  - reports/figures/training/fig3_2_densenet121_accuracy_progression.png (.pdf)
  - reports/figures/training/fig3_3_densenet121_composite_training_dashboard.png (.pdf)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Publication Aesthetics Configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def setup_dir(base_dir):
    out_dir = os.path.join(base_dir, 'reports', 'figures', 'training')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# Empirical 15-Epoch Training History from the 22,356 CXR Dataset (T4 GPU Run)
HISTORY_DATA = {
    "epochs": list(range(1, 16)),
    "train_loss": [0.2998, 0.1547, 0.1166, 0.1056, 0.0882, 0.0726, 0.0713, 0.0692, 0.0600, 0.0554, 0.0522, 0.0312, 0.0285, 0.0277, 0.0277],
    "val_loss":   [0.1175, 0.2806, 0.0691, 0.1069, 0.0908, 0.0662, 0.1673, 0.0491, 0.0979, 0.0525, 0.0971, 0.0423, 0.0496, 0.0436, 0.0386],
    "val_acc":    [96.06,  86.49,  97.50,  96.15,  96.69,  97.05,  92.93,  98.84,  96.69,  98.30,  96.69,  98.75,  98.48,  98.57,  98.39],
    "val_recall": [96.09,  88.88,  97.49,  96.47,  96.43,  97.20,  94.05,  98.86,  96.89,  98.37,  96.79,  98.70,  98.45,  98.50,  98.40],
    "val_auc":    [0.9959, 0.9942, 0.9984, 0.9964, 0.9973, 0.9984, 0.9974, 0.9993, 0.9976, 0.9990, 0.9977, 0.9994, 0.9992, 0.9993, 0.9993],
    "saved_checkpoints": [1, 3, 6, 8, 12, 15]
}


def plot_fig3_1_loss_convergence(data, out_dir):
    """
    Figure 3.1: Epoch vs Cross-Entropy Loss Convergence (Matching demo asset image copy.png).
    """
    print("Generating Figure 3.1: Epoch vs Loss Convergence...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    
    # Plot Trajectories
    ax.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.2, markersize=5.5, label='Training Loss (Cross-Entropy)')
    ax.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.2, markersize=5.5, label='Validation Loss (Cross-Entropy)')
    
    # Mark Minimum Validation Loss
    min_val_idx = np.argmin(val_loss)
    min_epoch = epochs[min_val_idx]
    min_loss = val_loss[min_val_idx]
    
    ax.scatter([min_epoch], [min_loss], color='#27ae60', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Minimum Val Loss: {min_loss:.4f}\n(Epoch {min_epoch})',
                xy=(min_epoch, min_loss), xytext=(min_epoch - 3.8, min_loss + 0.055),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#27ae60", lw=1))
                
    ax.set_title("Figure 3.1: Training & Validation Loss Convergence (DenseNet-121 + CBAM)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.35)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_1_densenet121_loss_convergence.png")
    pdf_path = os.path.join(out_dir, "fig3_1_densenet121_loss_convergence.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_2_accuracy_progression(data, out_dir):
    """
    Figure 3.2: Epoch vs Validation Accuracy and Sensitivity (Matching demo asset image copy 10.png).
    """
    print("Generating Figure 3.2: Epoch vs Accuracy Progression...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = data["epochs"]
    val_acc = data["val_acc"]
    val_recall = data["val_recall"]
    
    ax.plot(epochs, val_acc, 'D-', color='#16a085', linewidth=2.2, markersize=5.5, label='Validation Accuracy (%)')
    ax.plot(epochs, val_recall, '^-', color='#8e44ad', linewidth=2.2, markersize=5.5, label='Validation Sensitivity / Recall (%)')
    
    # Annotate Peak Accuracy
    max_acc_idx = np.argmax(val_acc)
    peak_epoch = epochs[max_acc_idx]
    peak_acc = val_acc[max_acc_idx]
    
    ax.scatter([peak_epoch], [peak_acc], color='#d35400', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Peak Val Accuracy: {peak_acc:.2f}%\n(Epoch {peak_epoch})',
                xy=(peak_epoch, peak_acc), xytext=(peak_epoch - 3.5, peak_acc - 4.5),
                arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#a04000',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2e9", ec="#d35400", lw=1))
                
    ax.set_title("Figure 3.2: Epoch vs Validation Accuracy & Sensitivity Progression", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Diagnostic Metric (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(84, 101)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_2_densenet121_accuracy_progression.png")
    pdf_path = os.path.join(out_dir, "fig3_2_densenet121_accuracy_progression.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_3_composite_dashboard(data, out_dir):
    """
    Figure 3.3: Two-Panel Composite Training Dashboard (Matching image copy.png and image copy 10.png combined).
    """
    print("Generating Figure 3.3: Composite Training Dashboard...")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    val_acc = data["val_acc"]
    val_auc = data["val_auc"]
    
    # --- PANEL A: Loss Dynamics ---
    ax0.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.0, markersize=5, label='Training Loss')
    ax0.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.0, markersize=5, label='Validation Loss')
    
    # Checkpoint markers
    for chk in data["saved_checkpoints"]:
        ax0.axvline(x=chk, color='#27ae60', linestyle=':', alpha=0.35, linewidth=1.2)
    ax0.scatter([15], [val_loss[-1]], color='#27ae60', s=130, zorder=5, edgecolor='black', linewidth=1.2)
    ax0.annotate(f'Best Val Loss: {val_loss[-1]:.4f}',
                xy=(15, val_loss[-1]), xytext=(10.5, val_loss[-1] + 0.05),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.2, headwidth=5),
                fontsize=8.5, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60", lw=0.8))
                
    ax0.set_title("A: Cross-Entropy Loss Minimization", fontsize=11.5, fontweight='bold', pad=10)
    ax0.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax0.set_ylabel("Loss", fontsize=10, fontweight='bold')
    ax0.set_xticks(epochs)
    ax0.set_ylim(0, 0.35)
    ax0.grid(True, linestyle='--', alpha=0.4)
    ax0.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', fontsize=9)
    
    # --- PANEL B: Diagnostic Metric Progression ---
    ax1.plot(epochs, val_acc, 'D-', color='#16a085', linewidth=2.0, markersize=5, label='Val Accuracy (%)')
    # Plot AUC on secondary axis or scaled
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, val_auc, '^-', color='#8e44ad', linewidth=2.0, markersize=5, label='Val ROC-AUC')
    
    ax1.set_title("B: Accuracy & ROC-AUC Trajectory", fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=10, fontweight='bold', color='#16a085')
    ax1_twin.set_ylabel("Validation ROC-AUC", fontsize=10, fontweight='bold', color='#8e44ad')
    ax1.set_xticks(epochs)
    ax1.set_ylim(84, 100)
    ax1_twin.set_ylim(0.990, 1.000)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, facecolor='#f8f9fa', fontsize=9)
    
    plt.suptitle("Figure 3.3: Hybrid DenseNet-121 + CBAM Training Dynamics across 15 Epochs (22,356 CXR Dataset)",
                 fontsize=13, fontweight='bold', y=1.03)
                 
    png_path = os.path.join(out_dir, "fig3_3_densenet121_composite_training_dashboard.png")
    pdf_path = os.path.join(out_dir, "fig3_3_densenet121_composite_training_dashboard.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = setup_dir(repo_root)
    print("=" * 70)
    print("TRAINING CONVERGENCE FIGURE GENERATOR: DENSENET-121 + CBAM")
    print(f"Output Directory: {out_dir}")
    print("=" * 70)
    
    plot_fig3_1_loss_convergence(HISTORY_DATA, out_dir)
    plot_fig3_2_accuracy_progression(HISTORY_DATA, out_dir)
    plot_fig3_3_composite_dashboard(HISTORY_DATA, out_dir)
    
    print("\nAll training convergence figures generated successfully!")


if __name__ == "__main__":
    main()
