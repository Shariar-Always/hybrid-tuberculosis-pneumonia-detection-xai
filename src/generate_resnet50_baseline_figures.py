"""
Publication-Grade Figure Generation Script: ResNet-50 Baseline & CBAM Ablation
==============================================================================
Generates high-resolution (300+ DPI & Vector PDF) figures for:
  - Training dynamics and loss convergence for ResNet-50 Baseline (Figs 3.16, 3.17, 3.18)
  - Holdout cohort confusion matrix for ResNet-50 Baseline (Fig 4.2G)
  - ResNet-50 CBAM Ablation Study (Fig 6.1C Ablation Study, mapping image copy 5.png)
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
    h_base_path = os.path.join(base_dir, 'models', 'history_resnet50_baseline.json')
    m_base_path = os.path.join(base_dir, 'models', 'metrics_resnet50_baseline.json')
    m_hyb_path = os.path.join(base_dir, 'models', 'metrics_resnet50_hybrid.json')
    
    with open(h_base_path, 'r', encoding='utf-8') as f:
        history_base = json.load(f)
    with open(m_base_path, 'r', encoding='utf-8') as f:
        metrics_base = json.load(f)
    with open(m_hyb_path, 'r', encoding='utf-8') as f:
        metrics_hyb = json.load(f)
        
    return history_base, metrics_base, metrics_hyb


def plot_fig3_16_loss_convergence(history, out_dir):
    """
    Figure 3.16: Epoch vs Cross-Entropy Loss for ResNet-50 Baseline.
    """
    print("Generating Figure 3.16: ResNet-50 Baseline Epoch vs Loss Convergence...")
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
    ax.annotate(f'Minimum Val Loss: {min_loss:.4f}\n(Epoch {min_epoch} Optimal Checkpoint)',
                xy=(min_epoch, min_loss), xytext=(min_epoch - 3.8, min_loss + 0.05),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#1e8449',
                bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#27ae60", lw=1))
                
    ax.set_title("Figure 3.16: Training & Validation Loss Dynamics (ResNet-50 Baseline)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.35)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_16_resnet50_baseline_loss_convergence.png")
    pdf_path = os.path.join(out_dir, "fig3_16_resnet50_baseline_loss_convergence.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_17_accuracy_progression(history, out_dir):
    """
    Figure 3.17: Epoch vs Validation Accuracy and Sensitivity for ResNet-50 Baseline.
    """
    print("Generating Figure 3.17: ResNet-50 Baseline Epoch vs Accuracy Progression...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    epochs = list(range(1, len(history["val_acc"]) + 1))
    val_acc = [x * 100 for x in history["val_acc"]]
    val_sens = [x * 100 for x in history["val_sens"]]
    
    ax.plot(epochs, val_acc, '^-', color='#16a085', linewidth=2.2, markersize=6, label='Validation Accuracy (%)')
    ax.plot(epochs, val_sens, 'v--', color='#8e44ad', linewidth=2.2, markersize=6, label='Validation Sensitivity / Recall (%)')
    
    # Mark peak accuracy
    max_acc_idx = np.argmax(val_acc)
    max_epoch = epochs[max_acc_idx]
    max_acc = val_acc[max_acc_idx]
    
    ax.scatter([max_epoch], [max_acc], color='#d35400', s=160, zorder=5, edgecolor='black', linewidth=1.5)
    ax.annotate(f'Peak Val Accuracy: {max_acc:.2f}%\n(Epoch {max_epoch})',
                xy=(max_epoch, max_acc), xytext=(max_epoch - 3.2, max_acc - 2.0),
                arrowprops=dict(facecolor='#d35400', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#ba4a00',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2e9", ec="#d35400", lw=1))
                
    ax.set_title("Figure 3.17: Validation Accuracy & Sensitivity Progression (ResNet-50 Baseline)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Metric Score (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(epochs)
    ax.set_ylim(95, 100)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888')
    ax.legend(loc='lower right', frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig3_17_resnet50_baseline_accuracy_progression.png")
    pdf_path = os.path.join(out_dir, "fig3_17_resnet50_baseline_accuracy_progression.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig3_18_composite_dashboard(history, out_dir):
    """
    Figure 3.18: Dual-Panel Composite Training Convergence Dashboard.
    """
    print("Generating Figure 3.18: ResNet-50 Baseline Dual-Panel Dashboard...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = [x * 100 for x in history["val_acc"]]
    val_auc = history["val_auc"]
    
    # Panel A: Loss Minimization
    ax1.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2.0, markersize=5, label='Training Loss')
    ax1.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.0, markersize=5, label='Validation Loss')
    min_idx = np.argmin(val_loss)
    ax1.scatter([epochs[min_idx]], [val_loss[min_idx]], color='#27ae60', s=130, zorder=5, edgecolor='black')
    ax1.annotate(f'Best Val Loss: {val_loss[min_idx]:.4f}\n(Epoch {epochs[min_idx]})',
                 xy=(epochs[min_idx], val_loss[min_idx]), xytext=(epochs[min_idx] - 3.5, val_loss[min_idx] + 0.05),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.2, headwidth=5),
                 fontsize=8.5, fontweight='bold', color='#1e8449',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60", lw=0.8))
    ax1.set_title("(A) Cross-Entropy Loss Minimization Trajectory", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=10, fontweight='bold')
    ax1.set_xticks(epochs)
    ax1.set_ylim(0, 0.35)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # Panel B: Dual-Axis Accuracy & ROC-AUC Progression
    color_acc = '#16a085'
    ax2.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Validation Accuracy (%)", color=color_acc, fontsize=10, fontweight='bold')
    line1 = ax2.plot(epochs, val_acc, '^-', color=color_acc, linewidth=2.0, markersize=5, label='Validation Accuracy (%)')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(95, 100)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    ax2_twin = ax2.twinx()
    color_auc = '#8e44ad'
    ax2_twin.set_ylabel("Validation ROC-AUC", color=color_auc, fontsize=10, fontweight='bold')
    line2 = ax2_twin.plot(epochs, val_auc, 'd--', color=color_auc, linewidth=2.0, markersize=5, label='Validation ROC-AUC')
    ax2_twin.tick_params(axis='y', labelcolor=color_auc)
    ax2_twin.set_ylim(0.995, 1.000)
    ax2_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    
    # Combine Legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='lower right', frameon=True, fontsize=9)
    ax2.set_title("(B) Validation Generalization & Discriminative Power", fontsize=11, fontweight='bold', pad=10)
    
    fig.suptitle("Figure 3.18: ResNet-50 Baseline Optimization & Diagnostic Metrics Dashboard", 
                 fontsize=12.5, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    png_path = os.path.join(out_dir, "fig3_18_resnet50_baseline_composite_dashboard.png")
    pdf_path = os.path.join(out_dir, "fig3_18_resnet50_baseline_composite_dashboard.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig4_2g_confusion_matrix(metrics, out_dir):
    """
    Figure 4.2G: ResNet-50 Baseline 3x3 Confusion Matrix.
    """
    print("Generating Figure 4.2G: ResNet-50 Baseline 3x3 Confusion Matrix...")
    cm = np.array(metrics['confusion_matrix'])
    classes = ['Normal', 'Tuberculosis', 'Pneumonia']
    
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=classes, yticklabels=classes,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                linewidths=1.2, linecolor='#dddddd', ax=ax)
                
    # Add row-normalized percentages in annotations
    for i in range(3):
        row_total = cm[i].sum()
        for j in range(3):
            val = cm[i, j]
            pct = (val / row_total) * 100
            ax.text(j + 0.5, i + 0.72, f"({pct:.1f}%)",
                    ha='center', va='center',
                    fontsize=9.5, color='#ffffff' if val > 400 else '#333333',
                    fontweight='normal')
                    
    ax.set_title("Figure 4.2G: Test Cohort Confusion Matrix (ResNet-50 Baseline)\nOverall Accuracy: 99.24% | Macro Sensitivity: 99.25% | Macro ROC-AUC: 0.9998",
                 fontsize=11, fontweight='bold', pad=14)
    ax.set_xlabel("Predicted Pathology", fontsize=10.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Gold-Standard Pathology", fontsize=10.5, fontweight='bold', labelpad=10)
    
    png_path = os.path.join(out_dir, "fig4_2g_resnet50_baseline_confusion_matrix.png")
    pdf_path = os.path.join(out_dir, "fig4_2g_resnet50_baseline_confusion_matrix.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def plot_fig6_1c_cbam_ablation(metrics_base, metrics_hyb, out_dir):
    """
    Figure 6.1C: ResNet-50 CBAM Ablation Study (Baseline vs. Hybrid).
    """
    print("Generating Figure 6.1C: ResNet-50 CBAM Ablation Study...")
    
    categories = ['Accuracy', 'Sensitivity\n(Macro Recall)', 'Specificity', 'TB Recall\n(Sensitivity)', 'Pneumonia\nRecall']
    
    cm_base = np.array(metrics_base['confusion_matrix'])
    cm_hyb = np.array(metrics_hyb['confusion_matrix'])
    
    base_acc = metrics_base['test_accuracy'] * 100
    base_sens = metrics_base['test_sensitivity'] * 100
    base_spec = metrics_base['test_specificity'] * 100
    base_tb_rec = (cm_base[1, 1] / cm_base[1].sum()) * 100
    base_pne_rec = (cm_base[2, 2] / cm_base[2].sum()) * 100
    
    hyb_acc = metrics_hyb['test_accuracy'] * 100
    hyb_sens = metrics_hyb['test_sensitivity'] * 100
    hyb_spec = metrics_hyb['test_specificity'] * 100
    hyb_tb_rec = (cm_hyb[1, 1] / cm_hyb[1].sum()) * 100
    hyb_pne_rec = (cm_hyb[2, 2] / cm_hyb[2].sum()) * 100
    
    base_scores = [base_acc, base_sens, base_spec, base_tb_rec, base_pne_rec]
    hyb_scores = [hyb_acc, hyb_sens, hyb_spec, hyb_tb_rec, hyb_pne_rec]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_scores, width, label='ResNet-50 Baseline (No Attention)', color='#7f8c8d', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, hyb_scores, width, label='ResNet-50 + CBAM (Hybrid)', color='#27ae60', edgecolor='black', linewidth=0.8)
    
    # Annotate values and delta
    for i in range(len(categories)):
        b_val = base_scores[i]
        h_val = hyb_scores[i]
        diff = h_val - b_val
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        diff_color = "#27ae60" if diff >= 0 else "#c0392b"
        
        ax.annotate(f"{b_val:.2f}%", (x[i] - width/2, b_val + 0.15), ha='center', fontsize=8.5, fontweight='bold', color='#333333')
        ax.annotate(f"{h_val:.2f}%", (x[i] + width/2, h_val + 0.15), ha='center', fontsize=8.5, fontweight='bold', color='#145a32')
        
        # Delta badge
        ax.annotate(f"Δ {diff_str}", (x[i], max(b_val, h_val) + 0.55), ha='center',
                    fontsize=8.5, fontweight='bold', color=diff_color,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#f8f9fa", ec=diff_color, lw=1))
                    
    ax.set_title("Figure 6.1C: Attention Ablation Analysis: ResNet-50 Baseline vs. Hybrid (+CBAM)\nHoldout Cohort Evaluation (N = 2,236 Independent Digital Radiographs)",
                 fontsize=11.5, fontweight='bold', pad=14)
    ax.set_ylabel("Diagnostic Score (%)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax.set_ylim(97.5, 100.8)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9.5)
    
    png_path = os.path.join(out_dir, "fig6_1c_cbam_ablation_resnet50.png")
    pdf_path = os.path.join(out_dir, "fig6_1c_cbam_ablation_resnet50.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def main():
    base_dir = r"E:\Git All Repo\hybrid-tuberculosis-pneumonia-detection-xai"
    train_dir, eval_dir = setup_dirs(base_dir)
    history, metrics_base, metrics_hyb = load_data(base_dir)
    
    # Generate Training Dynamics Figures
    plot_fig3_16_loss_convergence(history, train_dir)
    plot_fig3_17_accuracy_progression(history, train_dir)
    plot_fig3_18_composite_dashboard(history, train_dir)
    
    # Generate Evaluation Figures
    plot_fig4_2g_confusion_matrix(metrics_base, eval_dir)
    plot_fig6_1c_cbam_ablation(metrics_base, metrics_hyb, eval_dir)
    
    print("\nAll ResNet-50 Baseline figures generated successfully at 300+ DPI & Vector PDF!")


if __name__ == '__main__':
    main()
