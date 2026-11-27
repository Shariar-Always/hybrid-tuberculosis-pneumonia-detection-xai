"""
Publication-Grade Figure Generation Script: ResNet-50 + CBAM Hybrid Model
==========================================================================
Generates high-resolution (300+ DPI & Vector PDF) figures for:
  - Training dynamics and loss convergence (Figs 3.4, 3.5, 3.6)
  - Holdout cohort confusion matrix (Fig 4.2C)
  - Two-model comparative diagnostic benchmark (Fig 4.1 Preliminary)
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


def setup_dirs(base_dir):
    train_dir = os.path.join(base_dir, 'reports', 'figures', 'training')
    eval_dir = os.path.join(base_dir, 'reports', 'figures', 'eval')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return train_dir, eval_dir


def load_data(base_dir):
    hist_path = os.path.join(base_dir, 'models', 'history_resnet50_hybrid.json')
    metrics_resnet_path = os.path.join(base_dir, 'models', 'metrics_resnet50_hybrid.json')
    metrics_dense_path = os.path.join(base_dir, 'models', 'metrics_densenet121_hybrid.json')
    
    with open(hist_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    with open(metrics_resnet_path, 'r', encoding='utf-8') as f:
        metrics_resnet = json.load(f)
    with open(metrics_dense_path, 'r', encoding='utf-8') as f:
        metrics_dense = json.load(f)
        
    return history, metrics_resnet, metrics_dense


def plot_fig3_4_loss_convergence(history, out_dir):
    """
    Figure 3.4: Epoch vs Cross-Entropy Loss for ResNet-50 + CBAM.
    """
    print("Generating Figure 3.4: ResNet-50 Epoch vs Loss Convergence...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    
    ax.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.2, markersize=5.5, label='Training Loss (Cross-Entropy)')
    ax.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.2, markersize=5.5, label='Validation Loss (Cross-Entropy)')
    
    # Mark Minimum Validation Loss (Epoch 12)
    min_val_idx = np.argmin(val_loss)
    min_epoch = epochs[min_val_idx]
    min_loss = val_loss[min_val_idx]
    
    ax.scatter([min_epoch], [min_loss], color='#27ae60', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Minimum Val Loss: {min_loss:.4f}\n(Epoch {min_epoch})',
                xy=(min_epoch, min_loss), xytext=(min_epoch - 3.8, min_loss + 0.12),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#27ae60", lw=1))
                
    # Annotate Learning Rate Adjustment / Transient Perturbation at Epoch 7
    ax.annotate('LR Scheduler Plateau Step\n& Transient Recovery',
                xy=(7, val_loss[6]), xytext=(7.5, val_loss[6] - 0.05),
                arrowprops=dict(facecolor='#e67e22', shrink=0.08, width=1.2, headwidth=5),
                fontsize=8.5, fontweight='bold', color='#b9770e',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7", ec="#f39c12", lw=0.8))
                
    ax.set_title("Figure 3.4: Training & Validation Loss Dynamics (ResNet-50 + CBAM)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(-0.02, 0.80)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_4_resnet50_loss_convergence.png")
    pdf_path = os.path.join(out_dir, "fig3_4_resnet50_loss_convergence.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_5_accuracy_progression(history, out_dir):
    """
    Figure 3.5: Epoch vs Validation Accuracy and Sensitivity for ResNet-50 + CBAM.
    """
    print("Generating Figure 3.5: ResNet-50 Epoch vs Accuracy Progression...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = list(range(1, len(history["val_acc"]) + 1))
    val_acc = [x * 100 for x in history["val_acc"]]
    val_sens = [x * 100 for x in history["val_sens"]]
    
    ax.plot(epochs, val_acc, 'D-', color='#16a085', linewidth=2.2, markersize=5.5, label='Validation Accuracy (%)')
    ax.plot(epochs, val_sens, '^-', color='#8e44ad', linewidth=2.2, markersize=5.5, label='Validation Sensitivity / Recall (%)')
    
    # Annotate Peak Accuracy (Epoch 13)
    max_acc_idx = np.argmax(val_acc)
    peak_epoch = epochs[max_acc_idx]
    peak_acc = val_acc[max_acc_idx]
    
    ax.scatter([peak_epoch], [peak_acc], color='#d35400', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Peak Val Accuracy: {peak_acc:.2f}%\n(Epoch {peak_epoch})',
                xy=(peak_epoch, peak_acc), xytext=(peak_epoch - 3.8, peak_acc - 6.0),
                arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#a04000',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2e9", ec="#d35400", lw=1))
                
    ax.set_title("Figure 3.5: Validation Diagnostic Progression (ResNet-50 + CBAM)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Diagnostic Metric (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(75, 102)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_5_resnet50_accuracy_progression.png")
    pdf_path = os.path.join(out_dir, "fig3_5_resnet50_accuracy_progression.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_6_composite_dashboard(history, out_dir):
    """
    Figure 3.6: Two-Panel Composite Training Dashboard for ResNet-50 + CBAM.
    """
    print("Generating Figure 3.6: ResNet-50 Composite Training Dashboard...")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = [x * 100 for x in history["val_acc"]]
    val_auc = history["val_auc"]
    
    # --- PANEL A: Loss Dynamics ---
    ax0.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.0, markersize=5, label='Training Loss')
    ax0.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.0, markersize=5, label='Validation Loss')
    
    # Checkpoints: Epochs 1, 2, 4, 5, 6, 8, 12
    saved_epochs = [1, 2, 4, 5, 6, 8, 12]
    for chk in saved_epochs:
        ax0.axvline(x=chk, color='#27ae60', linestyle=':', alpha=0.35, linewidth=1.2)
        
    ax0.scatter([12], [val_loss[11]], color='#27ae60', s=130, zorder=5, edgecolor='black', linewidth=1.2)
    ax0.annotate(f'Best Val Loss: {val_loss[11]:.4f}\n(Epoch 12)',
                xy=(12, val_loss[11]), xytext=(9.2, val_loss[11] + 0.15),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.2, headwidth=5),
                fontsize=8.5, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60", lw=0.8))
                
    ax0.set_title("A: Cross-Entropy Loss Minimization", fontsize=11.5, fontweight='bold', pad=10)
    ax0.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax0.set_ylabel("Loss", fontsize=10, fontweight='bold')
    ax0.set_xticks(epochs)
    ax0.set_ylim(-0.02, 0.80)
    ax0.grid(True, linestyle='--', alpha=0.4)
    ax0.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', fontsize=9)
    
    # --- PANEL B: Diagnostic Metric Progression ---
    ax1.plot(epochs, val_acc, 'D-', color='#16a085', linewidth=2.0, markersize=5, label='Val Accuracy (%)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, val_auc, '^-', color='#8e44ad', linewidth=2.0, markersize=5, label='Val ROC-AUC')
    
    ax1.set_title("B: Accuracy & ROC-AUC Trajectory", fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=10, fontweight='bold', color='#16a085')
    ax1_twin.set_ylabel("Validation ROC-AUC", fontsize=10, fontweight='bold', color='#8e44ad')
    ax1.set_xticks(epochs)
    ax1.set_ylim(75, 102)
    ax1_twin.set_ylim(0.980, 1.002)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, facecolor='#f8f9fa', fontsize=9)
    
    plt.suptitle("Figure 3.6: Hybrid ResNet-50 + CBAM Training Dynamics across 15 Epochs (22,356 CXR Dataset)",
                 fontsize=13, fontweight='bold', y=1.03)
                 
    png_path = os.path.join(out_dir, "fig3_6_resnet50_composite_training_dashboard.png")
    pdf_path = os.path.join(out_dir, "fig3_6_resnet50_composite_training_dashboard.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig4_2c_confusion_matrix(metrics_resnet, out_dir):
    """
    Figure 4.2C: 3x3 Confusion Matrix for ResNet-50 + CBAM.
    """
    print("Generating Figure 4.2C: ResNet-50 Confusion Matrix...")
    cm = np.array(metrics_resnet["confusion_matrix"])
    classes = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA']
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot_labels = np.empty_like(cm, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot_labels[r, c] = f"{cm[r, c]:,}\n({cm_norm[r, c]:.1f}%)"
            
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    sns.heatmap(
        cm_norm,
        annot=annot_labels,
        fmt='',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Normalized Sensitivity / Recall (%)'},
        linewidths=1.2,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    acc = metrics_resnet["test_accuracy"] * 100
    auc = metrics_resnet["test_auc"]
    sens = metrics_resnet["test_sensitivity"] * 100
    spec = metrics_resnet["test_specificity"] * 100
    total_test = cm.sum()
    total_correct = np.trace(cm)
    
    ax.set_title(
        f"Figure 4.2C: Primary 3x3 Test Confusion Matrix (ResNet-50 + CBAM)\n"
        f"Test Accuracy: {acc:.2f}% | Sensitivity: {sens:.2f}% | Specificity: {spec:.2f}% | AUC: {auc:.4f}\n"
        f"(N = {total_test:,} Scans | Correctly Classified: {total_correct:,} / {total_test:,})",
        fontsize=11.5, fontweight='bold', pad=15, linespacing=1.3
    )
    
    ax.set_xlabel("Predicted Clinical Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Gold-Standard Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    png_path = os.path.join(out_dir, "fig4_2c_resnet50_confusion_matrix.png")
    pdf_path = os.path.join(out_dir, "fig4_2c_resnet50_confusion_matrix.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig4_1_twomodel_comparison(metrics_dense, metrics_resnet, out_dir):
    """
    Figure 4.1 Preliminary: Comparative Diagnostic Performance (DenseNet-121 vs ResNet-50).
    Directly maps demo asset image.png.
    """
    print("Generating Figure 4.1 Preliminary: Two-Model Diagnostic Comparison...")
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    
    metrics_names = ['Accuracy', 'Sensitivity\n(Recall)', 'Specificity', 'ROC-AUC']
    dense_vals = [
        metrics_dense['test_accuracy'] * 100,
        metrics_dense['test_sensitivity'] * 100,
        metrics_dense['test_specificity'] * 100,
        metrics_dense['test_auc'] * 100
    ]
    resnet_vals = [
        metrics_resnet['test_accuracy'] * 100,
        metrics_resnet['test_sensitivity'] * 100,
        metrics_resnet['test_specificity'] * 100,
        metrics_resnet['test_auc'] * 100
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.32
    
    rects1 = ax.bar(x - width/2, dense_vals, width, label='DenseNet-121 + CBAM', color='#3498db', edgecolor='#1b4f72', linewidth=1.2)
    rects2 = ax.bar(x + width/2, resnet_vals, width, label='ResNet-50 + CBAM', color='#2ecc71', edgecolor='#196f3d', linewidth=1.2)
    
    # Add numerical labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1b4f72')
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#196f3d')
                    
    ax.set_title("Figure 4.1: Comparative Diagnostic Performance on 2,236 Independent Test CXRs\nDenseNet-121 + CBAM vs. ResNet-50 + CBAM",
                 fontsize=11.5, fontweight='bold', pad=14)
    ax.set_ylabel("Diagnostic Score (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10, fontweight='bold')
    ax.set_ylim(95, 102)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig4_1_preliminary_twomodel_comparison.png")
    pdf_path = os.path.join(out_dir, "fig4_1_preliminary_twomodel_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_dir, eval_dir = setup_dirs(repo_root)
    history, metrics_resnet, metrics_dense = load_data(repo_root)
    
    print("=" * 75)
    print("PUBLICATION FIGURE GENERATOR: RESNET-50 + CBAM & TWO-MODEL BENCHMARK")
    print("=" * 75)
    
    plot_fig3_4_loss_convergence(history, train_dir)
    plot_fig3_5_accuracy_progression(history, train_dir)
    plot_fig3_6_composite_dashboard(history, train_dir)
    plot_fig4_2c_confusion_matrix(metrics_resnet, eval_dir)
    plot_fig4_1_twomodel_comparison(metrics_dense, metrics_resnet, eval_dir)
    
    print("\nAll ResNet-50 & Comparative figures generated successfully in 300+ DPI PNG & Vector PDF!")


if __name__ == "__main__":
    main()
