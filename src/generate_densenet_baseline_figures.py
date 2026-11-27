"""
Publication-Grade Figure Generation Script: DenseNet-121 Baseline & CBAM Ablation
==================================================================================
Generates high-resolution (300+ DPI & Vector PDF) figures for:
  - Training dynamics and loss convergence for DenseNet-121 Baseline (Figs 3.10, 3.11, 3.12)
  - Holdout cohort confusion matrix for DenseNet-121 Baseline (Fig 4.2E)
  - DenseNet-121 CBAM Ablation Study (Fig 6.1 Ablation Study, mapping image copy 5.png)
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
    h_base_path = os.path.join(base_dir, 'models', 'history_densenet121_baseline.json')
    m_base_path = os.path.join(base_dir, 'models', 'metrics_densenet121_baseline.json')
    m_hyb_path = os.path.join(base_dir, 'models', 'metrics_densenet121_hybrid.json')
    
    with open(h_base_path, 'r', encoding='utf-8') as f:
        history_base = json.load(f)
    with open(m_base_path, 'r', encoding='utf-8') as f:
        metrics_base = json.load(f)
    with open(m_hyb_path, 'r', encoding='utf-8') as f:
        metrics_hyb = json.load(f)
        
    return history_base, metrics_base, metrics_hyb


def plot_fig3_10_loss_convergence(history, out_dir):
    """
    Figure 3.10: Epoch vs Cross-Entropy Loss for DenseNet-121 Baseline.
    """
    print("Generating Figure 3.10: DenseNet-121 Baseline Epoch vs Loss Convergence...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    
    ax.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.2, markersize=5.5, label='Training Loss (Cross-Entropy)')
    ax.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.2, markersize=5.5, label='Validation Loss (Cross-Entropy)')
    
    # Mark Minimum Validation Loss (Epoch 14)
    min_val_idx = np.argmin(val_loss)
    min_epoch = epochs[min_val_idx]
    min_loss = val_loss[min_val_idx]
    
    ax.scatter([min_epoch], [min_loss], color='#27ae60', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Minimum Val Loss: {min_loss:.4f}\n(Epoch {min_epoch})',
                xy=(min_epoch, min_loss), xytext=(min_epoch - 3.8, min_loss + 0.055),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#27ae60", lw=1))
                
    ax.set_title("Figure 3.10: Training & Validation Loss Dynamics (DenseNet-121 Baseline)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.35)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_10_densenet121_baseline_loss_convergence.png")
    pdf_path = os.path.join(out_dir, "fig3_10_densenet121_baseline_loss_convergence.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_11_accuracy_progression(history, out_dir):
    """
    Figure 3.11: Epoch vs Validation Accuracy and Sensitivity for DenseNet-121 Baseline.
    """
    print("Generating Figure 3.11: DenseNet-121 Baseline Epoch vs Accuracy Progression...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = list(range(1, len(history["val_acc"]) + 1))
    val_acc = [x * 100 for x in history["val_acc"]]
    val_sens = [x * 100 for x in history["val_sens"]]
    
    ax.plot(epochs, val_acc, 'D-', color='#16a085', linewidth=2.2, markersize=5.5, label='Validation Accuracy (%)')
    ax.plot(epochs, val_sens, '^-', color='#8e44ad', linewidth=2.2, markersize=5.5, label='Validation Sensitivity / Recall (%)')
    
    # Annotate Peak Accuracy (Epoch 15)
    max_acc_idx = np.argmax(val_acc)
    peak_epoch = epochs[max_acc_idx]
    peak_acc = val_acc[max_acc_idx]
    
    ax.scatter([peak_epoch], [peak_acc], color='#d35400', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Peak Val Accuracy: {peak_acc:.2f}%\n(Epoch {peak_epoch})',
                xy=(peak_epoch, peak_acc), xytext=(peak_epoch - 3.8, peak_acc - 4.5),
                arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#a04000',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2e9", ec="#d35400", lw=1))
                
    ax.set_title("Figure 3.11: Validation Diagnostic Progression (DenseNet-121 Baseline)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Diagnostic Metric (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(92, 101)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_11_densenet121_baseline_accuracy_progression.png")
    pdf_path = os.path.join(out_dir, "fig3_11_densenet121_baseline_accuracy_progression.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_12_composite_dashboard(history, out_dir):
    """
    Figure 3.12: Two-Panel Composite Training Dashboard for DenseNet-121 Baseline.
    """
    print("Generating Figure 3.12: DenseNet-121 Baseline Composite Training Dashboard...")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = [x * 100 for x in history["val_acc"]]
    val_auc = history["val_auc"]
    
    # --- PANEL A: Loss Dynamics ---
    ax0.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.0, markersize=5, label='Training Loss')
    ax0.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.0, markersize=5, label='Validation Loss')
    
    # Saved Checkpoints: Epochs 1, 2, 4, 8, 10, 14
    saved_epochs = [1, 2, 4, 8, 10, 14]
    for chk in saved_epochs:
        ax0.axvline(x=chk, color='#27ae60', linestyle=':', alpha=0.35, linewidth=1.2)
        
    ax0.scatter([14], [val_loss[13]], color='#27ae60', s=130, zorder=5, edgecolor='black', linewidth=1.2)
    ax0.annotate(f'Best Val Loss: {val_loss[13]:.4f}\n(Epoch 14)',
                xy=(14, val_loss[13]), xytext=(10.5, val_loss[13] + 0.06),
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
    ax1.set_ylim(92, 101)
    ax1_twin.set_ylim(0.985, 1.001)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, facecolor='#f8f9fa', fontsize=9)
    
    plt.suptitle("Figure 3.12: Baseline DenseNet-121 (Without CBAM) Training Dynamics across 15 Epochs",
                 fontsize=13, fontweight='bold', y=1.03)
                 
    png_path = os.path.join(out_dir, "fig3_12_densenet121_baseline_composite_dashboard.png")
    pdf_path = os.path.join(out_dir, "fig3_12_densenet121_baseline_composite_dashboard.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig4_2e_confusion_matrix(metrics_base, out_dir):
    """
    Figure 4.2E: 3x3 Confusion Matrix for DenseNet-121 Baseline.
    """
    print("Generating Figure 4.2E: DenseNet-121 Baseline Confusion Matrix...")
    cm = np.array(metrics_base["confusion_matrix"])
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
    
    acc = metrics_base["test_accuracy"] * 100
    auc = metrics_base["test_auc"]
    sens = metrics_base["test_sensitivity"] * 100
    spec = metrics_base["test_specificity"] * 100
    total_test = cm.sum()
    total_correct = np.trace(cm)
    
    ax.set_title(
        f"Figure 4.2E: Primary 3x3 Test Confusion Matrix (DenseNet-121 Baseline)\n"
        f"Test Accuracy: {acc:.2f}% | Sensitivity: {sens:.2f}% | Specificity: {spec:.2f}% | AUC: {auc:.4f}\n"
        f"(N = {total_test:,} Scans | Correctly Classified: {total_correct:,} / {total_test:,})",
        fontsize=11.5, fontweight='bold', pad=15, linespacing=1.3
    )
    
    ax.set_xlabel("Predicted Clinical Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Gold-Standard Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    png_path = os.path.join(out_dir, "fig4_2e_densenet121_baseline_confusion_matrix.png")
    pdf_path = os.path.join(out_dir, "fig4_2e_densenet121_baseline_confusion_matrix.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig6_1_ablation_densenet121(metrics_base, metrics_hyb, out_dir):
    """
    Figure 6.1: CBAM Ablation Study (DenseNet-121 Baseline vs DenseNet-121 + CBAM).
    Directly maps demo asset image copy 5.png.
    """
    print("Generating Figure 6.1: DenseNet-121 CBAM Ablation Study (Mapping image copy 5.png)...")
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    
    categories = ['Accuracy', 'Sensitivity\n(Recall)', 'Specificity', 'ROC-AUC']
    base_scores = [
        metrics_base['test_accuracy'] * 100,
        metrics_base['test_sensitivity'] * 100,
        metrics_base['test_specificity'] * 100,
        metrics_base['test_auc'] * 100
    ]
    hyb_scores = [
        metrics_hyb['test_accuracy'] * 100,
        metrics_hyb['test_sensitivity'] * 100,
        metrics_hyb['test_specificity'] * 100,
        metrics_hyb['test_auc'] * 100
    ]
    
    x = np.arange(len(categories))
    width = 0.32
    
    rects1 = ax.bar(x - width/2, base_scores, width, label='DenseNet-121 Baseline (Without CBAM)', color='#7f8c8d', edgecolor='#2c3e50', linewidth=1.2)
    rects2 = ax.bar(x + width/2, hyb_scores, width, label='DenseNet-121 + CBAM (Proposed Hybrid)', color='#3498db', edgecolor='#1b4f72', linewidth=1.2)
    
    # Annotate values and delta gains
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2c3e50')
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1b4f72')
                    
    ax.set_title("Figure 6.1: CBAM Attention Ablation Benchmark on 2,236 Independent Test CXRs\nDenseNet-121 Baseline (No Attention) vs. DenseNet-121 + CBAM (Proposed)",
                 fontsize=11.5, fontweight='bold', pad=14)
    ax.set_ylabel("Diagnostic Score (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(96.5, 102)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig6_1_cbam_ablation_densenet121.png")
    pdf_path = os.path.join(out_dir, "fig6_1_cbam_ablation_densenet121.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_dir, eval_dir = setup_dirs(repo_root)
    history_base, metrics_base, metrics_hyb = load_data(repo_root)
    
    print("=" * 75)
    print("PUBLICATION FIGURE GENERATOR: DENSENET-121 BASELINE & ABLATION STUDY")
    print("=" * 75)
    
    plot_fig3_10_loss_convergence(history_base, train_dir)
    plot_fig3_11_accuracy_progression(history_base, train_dir)
    plot_fig3_12_composite_dashboard(history_base, train_dir)
    plot_fig4_2e_confusion_matrix(metrics_base, eval_dir)
    plot_fig6_1_ablation_densenet121(metrics_base, metrics_hyb, eval_dir)
    
    print("\nAll DenseNet-121 Baseline & Ablation figures generated successfully in 300+ DPI PNG & Vector PDF!")


if __name__ == "__main__":
    main()
