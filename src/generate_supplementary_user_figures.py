"""
Supplementary Publication-Grade Figures Generator
=================================================
This script generates two complementary publication-grade figures responding directly
to clinical literature standards and reference multi-panel visualizations:

1. Figure 3.19: All 6 Deep Learning Architectures Training & Validation Loss Convergence Grid (3x2 Panel)
   - Corresponds to multi-model comparative loss grids (e.g., InceptionV3 vs VGG19 vs CheXNet)
   - Evaluates DenseNet-121, ResNet-50, and EfficientNet-B0 (Hybrid vs Baseline)
   - Output: reports/figures/training/fig3_19_all_models_loss_grid.png (.pdf)

2. Figure 1.1b: Multi-Center Geographic Cohort Diagnostic Distribution Pie Charts
   - Corresponds to multi-country provenance pie charts (e.g., Moldova, Azerbaijan, Belarus, etc.)
   - Visualizes exact Normal, Tuberculosis, and Pneumonia distributions across the 8 international clinical hospital cohorts
   - Output: reports/figures/clinical_cohort_distribution/fig1_1b_geographic_cohort_distribution_pie.png (.pdf)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib publication aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def generate_all_models_loss_grid(base_dir):
    """
    Generates a 3x2 grid of training and validation loss curves across all six models.
    """
    models_info = [
        {
            "panel": "(A)",
            "title": "DenseNet-121 + CBAM (Hybrid)",
            "file": "history_densenet121_hybrid.json",
            "train_color": "#1b4f72",
            "val_color": "#e74c3c"
        },
        {
            "panel": "(B)",
            "title": "DenseNet-121 Baseline (No Attention)",
            "file": "history_densenet121_baseline.json",
            "train_color": "#2c3e50",
            "val_color": "#d35400"
        },
        {
            "panel": "(C)",
            "title": "ResNet-50 + CBAM (Diagnostic Champion)",
            "file": "history_resnet50_hybrid.json",
            "train_color": "#1b4f72",
            "val_color": "#c0392b"
        },
        {
            "panel": "(D)",
            "title": "ResNet-50 Baseline (No Attention)",
            "file": "history_resnet50_baseline.json",
            "train_color": "#34495e",
            "val_color": "#e67e22"
        },
        {
            "panel": "(E)",
            "title": "EfficientNet-B0 + CBAM (Edge Champion)",
            "file": "history_efficientnet_b0_hybrid.json",
            "train_color": "#1b4f72",
            "val_color": "#9b59b6"
        },
        {
            "panel": "(F)",
            "title": "EfficientNet-B0 Baseline (Early Stop Ep 10)",
            "file": "history_efficientnet_b0_baseline.json",
            "train_color": "#7f8c8d",
            "val_color": "#e67e22"
        }
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, info in enumerate(models_info):
        ax = axes[idx]
        file_path = os.path.join(base_dir, 'models', info['file'])
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        epochs = list(range(1, len(train_loss) + 1))
        
        # Plot curves
        ax.plot(epochs, train_loss, color=info['train_color'], linestyle='-', linewidth=2.2,
                marker='o', markersize=4.5, label='Training Loss', alpha=0.9)
        ax.plot(epochs, val_loss, color=info['val_color'], linestyle='--', linewidth=2.2,
                marker='s', markersize=4.5, label='Validation Loss', alpha=0.95)
        
        # Highlight minimum validation loss point
        min_idx = int(np.argmin(val_loss))
        min_val = val_loss[min_idx]
        ax.scatter([epochs[min_idx]], [min_val], color='#27ae60', s=70, zorder=5,
                   label=f'Min Val: {min_val:.4f} (Ep {epochs[min_idx]})')
        
        # Subplot styling
        ax.set_title(f"{info['panel']} {info['title']}", fontsize=12, fontweight='bold', pad=9)
        ax.set_xlabel('Epoch', fontsize=10.5, labelpad=5)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=10.5, labelpad=5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle=':', alpha=0.55, color='#95a5a6')
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.92, facecolor='white', edgecolor='#bdc3c7')
        
        # Soft background tint
        ax.set_facecolor('#fafbfc')

    plt.suptitle("Figure 3.19: Multi-Architecture Loss Convergence Profiles (Hybrid vs. Baseline)\nEvaluated on N = 22,356 Multi-Center CXR Cohorts (15 Epochs, AdamW Optimizer, CE Loss)",
                 fontsize=14.5, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    
    out_dir = os.path.join(base_dir, 'reports', 'figures', 'training')
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, 'fig3_19_all_models_loss_grid.png')
    pdf_path = os.path.join(out_dir, 'fig3_19_all_models_loss_grid.pdf')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"--> Saved Figure 3.19 (3x2 Loss Grid) to:\n    {png_path}\n    {pdf_path}")


def generate_cohort_pie_grid(base_dir):
    """
    Generates multi-center clinical cohort provenance pie charts matching reference Figure 4 style.
    """
    csv_path = os.path.join(base_dir, 'data', 'metadata.csv')
    df = pd.read_csv(csv_path)

    # Standardized clinical cohorts mapping
    cohorts = [
        {
            "name": "Guangzhou Women & Children's Med Ctr",
            "country": "China (East Asia)",
            "query": "Guangzhou Women and Children's Medical Center"
        },
        {
            "name": "TBX11K Multi-Center Consortium",
            "country": "China / International",
            "query": "TBX11K Multi-Center Screening Consortium"
        },
        {
            "name": "NIH Clinical Center & RSNA Challenge",
            "country": "Bethesda, USA (North America)",
            "query": "NIH Clinical Center / RSNA Pneumonia Challenge"
        },
        {
            "name": "NIAID TB Portals / Hamad Medical Corp",
            "country": "Doha, Qatar / Global",
            "query": "NIAID TB Portals & Qatar Univ / Univ of Dhaka (Hamad Medical Corp)"
        },
        {
            "name": "Regional Pulmonology Hospital",
            "country": "Rawalpindi / Islamabad (Pakistan)",
            "query": "Regional Pulmonology Hospital, Rawalpindi/Islamabad"
        },
        {
            "name": "Hamad General Hospital Cohort",
            "country": "Doha, Qatar (Middle East)",
            "query": "Qatar Univ & Univ of Dhaka / Hamad General Hospital"
        },
        {
            "name": "Shenzhen No. 3 People's Hospital",
            "country": "Guangdong, China (East Asia)",
            "query": "Shenzhen No. 3 People's Hospital, Guangdong Medical College"
        },
        {
            "name": "Belarus National TB Program & Montgomery",
            "country": "Belarus (Europe) & Maryland (USA)",
            "query": ["Belarus National Tuberculosis Surveillance Program", "Montgomery County TB Control Program (DHHS / NIH NLM)"]
        }
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
    axes = axes.flatten()

    colors = {
        'NORMAL': '#2ecc71',
        'TUBERCULOSIS': '#e74c3c',
        'PNEUMONIA': '#3498db'
    }

    panel_letters = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']

    for idx, cohort in enumerate(cohorts):
        ax = axes[idx]
        q = cohort['query']
        if isinstance(q, list):
            sub_df = df[df['hospital_name'].isin(q)]
        else:
            sub_df = df[df['hospital_name'] == q]
            
        counts = sub_df['class_name'].value_counts()
        total_n = len(sub_df)
        
        labels = [c for c in ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA'] if c in counts]
        sizes = [counts[c] for c in labels]
        c_list = [colors[c] for c in labels]
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct*total_n/100))})" if pct > 4 else '',
            pctdistance=0.6,
            startangle=140,
            colors=c_list,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5)
        )
        
        # Center circle text
        ax.text(0, 0, f"N =\n{total_n:,}", ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8.5)
            autotext.set_weight('bold')

        title_text = f"{panel_letters[idx]} {cohort['name']}\n[{cohort['country']}]"
        ax.set_title(title_text, fontsize=10, fontweight='bold', pad=10, color='#1a252f')

    # Shared legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Normal (Healthy Control)'),
        mpatches.Patch(color='#e74c3c', label='Active Pulmonary Tuberculosis (TB)'),
        mpatches.Patch(color='#3498db', label='Bacterial & Viral Pneumonia')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=11.5,
               framealpha=0.95, facecolor='white', edgecolor='#bdc3c7', bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("Figure 1.1B: International Clinical Hospital Cohort Provenance & Diagnostic Breakdown (N = 22,356)\nMulti-Center Distribution across 8 Healthcare Institutions in Asia, Europe, North America, and the Middle East",
                 fontsize=14.5, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_dir = os.path.join(base_dir, 'reports', 'figures', 'clinical_cohort_distribution')
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, 'fig1_1b_geographic_cohort_distribution_pie.png')
    pdf_path = os.path.join(out_dir, 'fig1_1b_geographic_cohort_distribution_pie.pdf')

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"--> Saved Figure 1.1B (Cohort Provenance Donut/Pie Grid) to:\n    {png_path}\n    {pdf_path}")


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"Executing Supplementary User Figures Generator with base directory: {base_dir}")
    generate_all_models_loss_grid(base_dir)
    generate_cohort_pie_grid(base_dir)
    print("--> Supplementary figures successfully generated!")
