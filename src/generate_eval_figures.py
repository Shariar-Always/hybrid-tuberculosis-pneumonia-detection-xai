"""
Publication-Grade Figure Generation Script: Confusion Matrix & Diagnostic Benchmarks
=====================================================================================
Directly maps demo asset 'image copy 2.png' to the champion model
DenseNet-121 + CBAM Hybrid evaluated on the 2,236 test cohort.

Outputs:
  - reports/figures/eval/fig4_2a_densenet121_champion_confusion_matrix.png (.pdf)
  - reports/figures/eval/CONFUSION_MATRIX_EVALUATION.md
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Publication Aesthetics Configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def setup_dir(base_dir):
    out_dir = os.path.join(base_dir, 'reports', 'figures', 'eval')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_fig4_2a_confusion_matrix(metrics_path, out_dir):
    print(f"Loading test metrics from: {metrics_path}")
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    cm = np.array(data["confusion_matrix"])
    print("Confusion Matrix:")
    print(cm)
    
    classes = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA']
    
    # Compute row-normalized percentages (True Class Recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Formulate dual text annotations: "Count\n(Percentage%)"
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
    
    # Header summary text
    acc = data["test_accuracy"] * 100
    auc = data["test_auc"]
    sens = data["test_sensitivity"] * 100
    spec = data["test_specificity"] * 100
    total_test = cm.sum()
    total_correct = np.trace(cm)
    
    ax.set_title(
        f"Figure 4.2A: Primary 3x3 Test Confusion Matrix (DenseNet-121 + CBAM)\n"
        f"Test Accuracy: {acc:.2f}% | Sensitivity: {sens:.2f}% | Specificity: {spec:.2f}% | AUC: {auc:.4f}\n"
        f"(N = {total_test:,} Scans | Correctly Classified: {total_correct:,} / {total_test:,})",
        fontsize=11.5, fontweight='bold', pad=15, linespacing=1.3
    )
    
    ax.set_xlabel("Predicted Clinical Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Gold-Standard Pathology", fontsize=11, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    png_path = os.path.join(out_dir, "fig4_2a_densenet121_champion_confusion_matrix.png")
    pdf_path = os.path.join(out_dir, "fig4_2a_densenet121_champion_confusion_matrix.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = setup_dir(repo_root)
    metrics_path = os.path.join(repo_root, 'models', 'metrics_densenet121_hybrid.json')
    
    print("=" * 70)
    print("CONFUSION MATRIX FIGURE GENERATOR: DENSENET-121 + CBAM")
    print("=" * 70)
    
    plot_fig4_2a_confusion_matrix(metrics_path, out_dir)
    print("\nConfusion Matrix figures generated successfully in 300+ DPI PNG and Vector PDF!")


if __name__ == "__main__":
    main()
