"""
Publication-Grade Figure Generation Script: EfficientNet-B0 + CBAM & Tri-Model Benchmark
========================================================================================
Generates high-resolution (300+ DPI & Vector PDF) figures for:
  - Training dynamics and loss convergence for EfficientNet-B0 + CBAM (Figs 3.7, 3.8, 3.9)
  - Holdout cohort confusion matrix for EfficientNet-B0 + CBAM (Fig 4.2D)
  - Comprehensive 3-Model Hybrid Comparative Diagnostic Benchmark (Fig 4.1 Tri-Model)
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


def load_all_hybrid_data(base_dir):
    h_eff_path = os.path.join(base_dir, 'models', 'history_efficientnet_b0_hybrid.json')
    m_eff_path = os.path.join(base_dir, 'models', 'metrics_efficientnet_b0_hybrid.json')
    m_res_path = os.path.join(base_dir, 'models', 'metrics_resnet50_hybrid.json')
    m_den_path = os.path.join(base_dir, 'models', 'metrics_densenet121_hybrid.json')
    
    with open(h_eff_path, 'r', encoding='utf-8') as f:
        history_eff = json.load(f)
    with open(m_eff_path, 'r', encoding='utf-8') as f:
        metrics_eff = json.load(f)
    with open(m_res_path, 'r', encoding='utf-8') as f:
        metrics_res = json.load(f)
    with open(m_den_path, 'r', encoding='utf-8') as f:
        metrics_den = json.load(f)
        
    return history_eff, metrics_eff, metrics_res, metrics_den


def plot_fig3_7_loss_convergence(history, out_dir):
    """
    Figure 3.7: Epoch vs Cross-Entropy Loss for EfficientNet-B0 + CBAM.
    """
    print("Generating Figure 3.7: EfficientNet-B0 Epoch vs Loss Convergence...")
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
                xy=(min_epoch, min_loss), xytext=(min_epoch - 3.8, min_loss + 0.055),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#27ae60", lw=1))
                
    ax.set_title("Figure 3.7: Training & Validation Loss Dynamics (EfficientNet-B0 + CBAM)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.35)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_7_efficientnet_loss_convergence.png")
    pdf_path = os.path.join(out_dir, "fig3_7_efficientnet_loss_convergence.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_8_accuracy_progression(history, out_dir):
    """
    Figure 3.8: Epoch vs Validation Accuracy and Sensitivity for EfficientNet-B0 + CBAM.
    """
    print("Generating Figure 3.8: EfficientNet-B0 Epoch vs Accuracy Progression...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = list(range(1, len(history["val_acc"]) + 1))
    val_acc = [x * 100 for x in history["val_acc"]]
    val_sens = [x * 100 for x in history["val_sens"]]
    
    ax.plot(epochs, val_acc, 'D-', color='#16a085', linewidth=2.2, markersize=5.5, label='Validation Accuracy (%)')
    ax.plot(epochs, val_sens, '^-', color='#8e44ad', linewidth=2.2, markersize=5.5, label='Validation Sensitivity / Recall (%)')
    
    # Annotate Peak Accuracy (Epoch 10 & 12)
    max_acc_idx = np.argmax(val_acc)
    peak_epoch = epochs[max_acc_idx]
    peak_acc = val_acc[max_acc_idx]
    
    ax.scatter([peak_epoch], [peak_acc], color='#d35400', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Peak Val Accuracy: {peak_acc:.2f}%\n(Epoch {peak_epoch})',
                xy=(peak_epoch, peak_acc), xytext=(peak_epoch - 3.8, peak_acc - 4.5),
                arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#a04000',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2e9", ec="#d35400", lw=1))
                
    ax.set_title("Figure 3.8: Validation Diagnostic Progression (EfficientNet-B0 + CBAM)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Diagnostic Metric (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(85, 101)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_8_efficientnet_accuracy_progression.png")
    pdf_path = os.path.join(out_dir, "fig3_8_efficientnet_accuracy_progression.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_9_composite_dashboard(history, out_dir):
    """
    Figure 3.9: Two-Panel Composite Training Dashboard for EfficientNet-B0 + CBAM.
    """
    print("Generating Figure 3.9: EfficientNet-B0 Composite Training Dashboard...")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = [x * 100 for x in history["val_acc"]]
    val_auc = history["val_auc"]
    
    # --- PANEL A: Loss Dynamics ---
    ax0.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.0, markersize=5, label='Training Loss')
    ax0.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.0, markersize=5, label='Validation Loss')
    
    # Saved Checkpoints: Epochs 1, 2, 3, 4, 5, 8, 12
    saved_epochs = [1, 2, 3, 4, 5, 8, 12]
    for chk in saved_epochs:
        ax0.axvline(x=chk, color='#27ae60', linestyle=':', alpha=0.35, linewidth=1.2)
        
    ax0.scatter([12], [val_loss[11]], color='#27ae60', s=130, zorder=5, edgecolor='black', linewidth=1.2)
    ax0.annotate(f'Best Val Loss: {val_loss[11]:.4f}\n(Epoch 12)',
                xy=(12, val_loss[11]), xytext=(9.2, val_loss[11] + 0.06),
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
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, val_auc, '^-', color='#8e44ad', linewidth=2.0, markersize=5, label='Val ROC-AUC')
    
    ax1.set_title("B: Accuracy & ROC-AUC Trajectory", fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=10, fontweight='bold', color='#16a085')
    ax1_twin.set_ylabel("Validation ROC-AUC", fontsize=10, fontweight='bold', color='#8e44ad')
    ax1.set_xticks(epochs)
    ax1.set_ylim(85, 101)
    ax1_twin.set_ylim(0.990, 1.001)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, facecolor='#f8f9fa', fontsize=9)
    
    plt.suptitle("Figure 3.9: Hybrid EfficientNet-B0 + CBAM Training Dynamics across 15 Epochs (22,356 CXR Dataset)",
                 fontsize=13, fontweight='bold', y=1.03)
                 
    png_path = os.path.join(out_dir, "fig3_9_efficientnet_composite_training_dashboard.png")
    pdf_path = os.path.join(out_dir, "fig3_9_efficientnet_composite_training_dashboard.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig4_2d_confusion_matrix(metrics_eff, out_dir):
    """
    Figure 4.2D: 3x3 Confusion Matrix for EfficientNet-B0 + CBAM.
    """
    print("Generating Figure 4.2D: EfficientNet-B0 Confusion Matrix...")
    cm = np.array(metrics_eff["confusion_matrix"])
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
    
    acc = metrics_eff["test_accuracy"] * 100
    auc = metrics_eff["test_auc"]
    sens = metrics_eff["test_sensitivity"] * 100
    spec = metrics_eff["test_specificity"] * 100
    total_test = cm.sum()
    total_correct = np.trace(cm)
    
    ax.set_title(
        f"Figure 4.2D: Primary 3x3 Test Confusion Matrix (EfficientNet-B0 + CBAM)\n"
        f"Test Accuracy: {acc:.2f}% | Sensitivity: {sens:.2f}% | Specificity: {spec:.2f}% | AUC: {auc:.4f}\n"
        f"(N = {total_test:,} Scans | Correctly Classified: {total_correct:,} / {total_test:,})",
        fontsize=11.5, fontweight='bold', pad=15, linespacing=1.3
    )
    
    ax.set_xlabel("Predicted Clinical Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Gold-Standard Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    png_path = os.path.join(out_dir, "fig4_2d_efficientnet_confusion_matrix.png")
    pdf_path = os.path.join(out_dir, "fig4_2d_efficientnet_confusion_matrix.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig4_1_three_hybrid_models_comparison(metrics_den, metrics_res, metrics_eff, out_dir):
    """
    Figure 4.1 Master: Tri-Model Diagnostic Performance Comparison.
    Directly maps demo asset image.png.
    """
    print("Generating Figure 4.1 Master: Tri-Model Hybrid Diagnostic Comparison...")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    
    metrics_names = ['Accuracy', 'Sensitivity\n(Recall)', 'Specificity', 'ROC-AUC']
    den_vals = [
        metrics_den['test_accuracy'] * 100,
        metrics_den['test_sensitivity'] * 100,
        metrics_den['test_specificity'] * 100,
        metrics_den['test_auc'] * 100
    ]
    res_vals = [
        metrics_res['test_accuracy'] * 100,
        metrics_res['test_sensitivity'] * 100,
        metrics_res['test_specificity'] * 100,
        metrics_res['test_auc'] * 100
    ]
    eff_vals = [
        metrics_eff['test_accuracy'] * 100,
        metrics_eff['test_sensitivity'] * 100,
        metrics_eff['test_specificity'] * 100,
        metrics_eff['test_auc'] * 100
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    rects1 = ax.bar(x - width, den_vals, width, label='DenseNet-121 + CBAM (30.0 MB)', color='#3498db', edgecolor='#1b4f72', linewidth=1.2)
    rects2 = ax.bar(x, res_vals, width, label='ResNet-50 + CBAM (98.6 MB)', color='#2ecc71', edgecolor='#196f3d', linewidth=1.2)
    rects3 = ax.bar(x + width, eff_vals, width, label='EfficientNet-B0 + CBAM (18.5 MB)', color='#e67e22', edgecolor='#935116', linewidth=1.2)
    
    # Value annotations on bars
    for rects, col in [(rects1, '#1b4f72'), (rects2, '#196f3d'), (rects3, '#935116')]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=col)
                        
    ax.set_title("Figure 4.1: Comparative Diagnostic Performance on 2,236 Independent Test CXRs\nDenseNet-121 + CBAM vs. ResNet-50 + CBAM vs. EfficientNet-B0 + CBAM",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel("Diagnostic Score (%)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10.5, fontweight='bold')
    ax.set_ylim(95.5, 102)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig4_1_three_hybrid_models_comparison.png")
    pdf_path = os.path.join(out_dir, "fig4_1_three_hybrid_models_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_dir, eval_dir = setup_dirs(repo_root)
    history_eff, metrics_eff, metrics_res, metrics_den = load_all_hybrid_data(repo_root)
    
    print("=" * 80)
    print("PUBLICATION FIGURE GENERATOR: EFFICIENTNET-B0 & TRI-MODEL HYBRID BENCHMARK")
    print("=" * 80)
    
    plot_fig3_7_loss_convergence(history_eff, train_dir)
    plot_fig3_8_accuracy_progression(history_eff, train_dir)
    plot_fig3_9_composite_dashboard(history_eff, train_dir)
    plot_fig4_2d_confusion_matrix(metrics_eff, eval_dir)
    plot_fig4_1_three_hybrid_models_comparison(metrics_den, metrics_res, metrics_eff, eval_dir)
    
    print("\nAll EfficientNet-B0 & Tri-Model figures generated successfully in 300+ DPI PNG & Vector PDF!")


if __name__ == "__main__":
    main()
