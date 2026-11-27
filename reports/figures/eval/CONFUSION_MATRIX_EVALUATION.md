# Confusion Matrix & Diagnostic Discrimination Analysis
## Evaluation of Hybrid Architectures on 2,236 Multi-Center Clinical Scans

**Directory:** `reports/figures/eval/`  
**Model Architectures Evaluated:** 
- DenseNet-121 + CBAM Hybrid (`fig4_2a`)
- ResNet-50 + CBAM Hybrid (`fig4_2c`)
- EfficientNet-B0 + CBAM Hybrid (`fig4_2d`)
- DenseNet-121 Baseline No CBAM (`fig4_2e`)
- DenseNet-121 CBAM Ablation Study (`fig6_1_cbam_ablation_densenet121`)
- Tri-Model Comparative Diagnostic Evaluation (`fig4_1_three_hybrid_models_comparison`)  
**Test Cohort Scale:** 2,236 Unique Digital Chest Radiographs (Hold-out Test Set)  
**Figure Standard:** 300+ DPI Raster PNG and Scale-Independent Vector PDF (IEEE / Nature / Elsevier Standards)  
**Associated Generator Scripts:** `src/generate_eval_figures.py`, `src/generate_resnet50_figures.py`, and `src/generate_efficientnet_figures.py`  

---

## 1. Executive Diagnostic Breakdown

The independent hold-out test cohort comprises **2,236 digital chest radiographs** curated from 8 multi-center clinical hospital cohorts (Normal: 929, Tuberculosis: 634, Pneumonia: 673). 

All three proposed hybrid architectures demonstrate outstanding diagnostic efficacy, with ResNet-50 + CBAM leading in overall accuracy (**99.42%**), followed by EfficientNet-B0 + CBAM (**98.79%**) and DenseNet-121 + CBAM (**98.57%**):

| Metric | DenseNet-121 + CBAM | ResNet-50 + CBAM | EfficientNet-B0 + CBAM | DenseNet-121 Baseline | Clinical Significance & Designation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Overall Test Accuracy** | 98.57% (2,204/2,236) | **99.42% (2,223/2,236)** | 98.79% (2,209/2,236) | 99.24% (2,219/2,236) | **ResNet-50:** Primary Diagnostic Champion |
| **Macro Sensitivity (Recall)** | 98.66% | **99.38%** | 98.70% | 99.27% | All models detect $>98.5\%$ of pulmonary pathology. |
| **Macro Specificity** | 99.29% | **99.68%** | 99.34% | 99.61% | Specificity exceeds $99.25\%$ across all architectures. |
| **Macro ROC-AUC** | 0.9997 | **0.9999** | 0.9996 | 0.9998 | All models achieve near-perfect ROC separation ($>0.999$). |
| **Total Test Errors** | 32 (1.43%) | **13 (0.58%)** | 27 (1.21%) | 17 (0.76%) | Negligible diagnostic error rates. |
| **Model Weights Size** | 30.0 MB | 98.6 MB | **18.5 MB** | 29.5 MB | **EfficientNet-B0: Ultra-Lightweight Mobile Edge Champion** |
| **Average Epoch Time** | 393.4s | 388.9s | **376.4s** | 397.1s | **EfficientNet-B0: Fastest Training Latency** |

---

## 2. Quantitative 3x3 Confusion Matrix Distributions

### A. EfficientNet-B0 + CBAM (Figure 4.2D)
```
========================================================================================
                          PREDICTED PATHOLOGY
========================================================================================
TRUE GOLD-STANDARD        NORMAL              TUBERCULOSIS        PNEUMONIA           TOTAL
----------------------------------------------------------------------------------------
NORMAL                    924 (99.46%)        2 (0.22%)           3 (0.32%)           929
TUBERCULOSIS              8 (1.26%)           625 (98.58%)        1 (0.16%)           634
PNEUMONIA                 13 (1.93%)          0 (0.00%)           660 (98.07%)        673
----------------------------------------------------------------------------------------
TOTAL PREDICTED           945                 627                 664                 2,236
========================================================================================
```

### B. ResNet-50 + CBAM (Figure 4.2C)
```
========================================================================================
                          PREDICTED PATHOLOGY
========================================================================================
TRUE GOLD-STANDARD        NORMAL              TUBERCULOSIS        PNEUMONIA           TOTAL
----------------------------------------------------------------------------------------
NORMAL                    926 (99.68%)        0 (0.00%)           3 (0.32%)           929
TUBERCULOSIS              7 (1.10%)           627 (98.90%)        0 (0.00%)           634
PNEUMONIA                 3 (0.45%)           0 (0.00%)           670 (99.55%)        673
----------------------------------------------------------------------------------------
TOTAL PREDICTED           936                 627                 673                 2,236
========================================================================================
```

### C. DenseNet-121 + CBAM (Figure 4.2A)
```
========================================================================================
                          PREDICTED PATHOLOGY
========================================================================================
TRUE GOLD-STANDARD        NORMAL              TUBERCULOSIS        PNEUMONIA           TOTAL
----------------------------------------------------------------------------------------
NORMAL                    909 (97.85%)        9 (0.97%)           11 (1.18%)          929
TUBERCULOSIS              7 (1.10%)           626 (98.74%)        1 (0.16%)           634
PNEUMONIA                 2 (0.30%)           2 (0.30%)           669 (99.41%)        673
----------------------------------------------------------------------------------------
TOTAL PREDICTED           918                 637                 681                 2,236
========================================================================================
```

### D. DenseNet-121 Baseline (No Attention) (Figure 4.2E)
```
========================================================================================
                          PREDICTED PATHOLOGY
========================================================================================
TRUE GOLD-STANDARD        NORMAL              TUBERCULOSIS        PNEUMONIA           TOTAL
----------------------------------------------------------------------------------------
NORMAL                    920 (99.03%)        6 (0.65%)           3 (0.32%)           929
TUBERCULOSIS              3 (0.47%)           631 (99.53%)        0 (0.00%)           634
PNEUMONIA                 4 (0.59%)           1 (0.15%)           668 (99.26%)        673
----------------------------------------------------------------------------------------
TOTAL PREDICTED           927                 638                 671                 2,236
========================================================================================
```

---

## 3. Directory Contents & Generated Assets

| Filename | Format | Dimensions / DPI | Description & Purpose |
| :--- | :--- | :--- | :--- |
| `fig4_2a_densenet121_champion_confusion_matrix.png` | Raster PNG | $2184 \times 1974$ (300 DPI) | Figure 4.2A: DenseNet-121 + CBAM 3x3 Confusion Matrix |
| `fig4_2a_densenet121_champion_confusion_matrix.pdf` | Vector PDF | Scale-Independent Vector | Vector format for DenseNet-121 confusion matrix |
| `fig4_2c_resnet50_confusion_matrix.png` | Raster PNG | $2340 \times 1980$ (300 DPI) | Figure 4.2C: ResNet-50 + CBAM 3x3 Confusion Matrix |
| `fig4_2c_resnet50_confusion_matrix.pdf` | Vector PDF | Scale-Independent Vector | Vector format for ResNet-50 confusion matrix |
| `fig4_2d_efficientnet_confusion_matrix.png` | Raster PNG | $2340 \times 1980$ (300 DPI) | Figure 4.2D: EfficientNet-B0 + CBAM 3x3 Confusion Matrix |
| `fig4_2d_efficientnet_confusion_matrix.pdf` | Vector PDF | Scale-Independent Vector | Vector format for EfficientNet-B0 confusion matrix |
| `fig4_1_three_hybrid_models_comparison.png` | Raster PNG | $3450 \times 1860$ (300 DPI) | Figure 4.1 Master: Tri-Model Hybrid Diagnostic Comparison Bar Chart |
| `fig4_1_three_hybrid_models_comparison.pdf` | Vector PDF | Scale-Independent Vector | Vector format for master comparative performance bar chart |
| `fig4_2e_densenet121_baseline_confusion_matrix.png` | Raster PNG | $2340 \times 1980$ (300 DPI) | Figure 4.2E: DenseNet-121 Baseline 3x3 Confusion Matrix |
| `fig4_2e_densenet121_baseline_confusion_matrix.pdf` | Vector PDF | Scale-Independent Vector | Vector format for DenseNet-121 baseline confusion matrix |
| `fig6_1_cbam_ablation_densenet121.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 6.1: CBAM Ablation Study for DenseNet-121 (Hybrid vs. Baseline) |
| `fig6_1_cbam_ablation_densenet121.pdf` | Vector PDF | Scale-Independent Vector | Vector format for CBAM ablation study |
| `CONFUSION_MATRIX_EVALUATION.md` | Markdown | Academic Documentation | Quantitative statistical breakdown and manuscript snippets |

---

## 4. Copy-Paste LaTeX Figure Code (For Journal Manuscripts)

### Figure 4.1 Master: Tri-Model Diagnostic Comparison Bar Chart
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{reports/figures/eval/fig4_1_three_hybrid_models_comparison.pdf}
\caption{\textbf{Comparative Diagnostic Performance of the Three Proposed Hybrid Architectures on 2,236 Multi-Center Clinical Radiographs.} 
Comparison across overall Accuracy, Macro Sensitivity, Macro Specificity, and ROC-AUC. ResNet-50 + CBAM delivers peak diagnostic accuracy ($99.42\%$), while EfficientNet-B0 + CBAM achieves $98.79\%$ accuracy with an ultra-compact $18.5\text{ MB}$ parameter footprint suitable for edge mobile deployment.}
\label{fig:trimodel_hybrid_comparison}
\end{figure*}
```

### Figure 4.2D: EfficientNet-B0 Confusion Matrix
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{reports/figures/eval/fig4_2d_efficientnet_confusion_matrix.pdf}
\caption{\textbf{Primary Multiclass Confusion Matrix for Hybrid EfficientNet-B0 + CBAM on the 2,236 Holdout Test Cohort.} 
Cells display absolute patient counts and row-normalized class sensitivities. The lightweight model achieves $98.79\%$ accuracy, $98.58\%$ sensitivity for Tuberculosis, and $98.07\%$ sensitivity for Pneumonia with only 27 total misclassifications out of 2,236 examinations.}
\label{fig:confusion_matrix_efficientnet}
\end{figure}
```

### Figure 4.2E: DenseNet-121 Baseline Confusion Matrix
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{reports/figures/eval/fig4_2e_densenet121_baseline_confusion_matrix.pdf}
\caption{\textbf{Confusion Matrix for DenseNet-121 Baseline (Without Attention) on the 2,236 Holdout Test Cohort.} 
DenseNet-121 baseline achieves $99.24\%$ overall accuracy with $99.53\%$ recall for active Tuberculosis ($631/634$ scans) and zero cross-confusion with Pneumonia.}
\label{fig:confusion_matrix_densenet_baseline}
\end{figure}
```

---

## 5. CBAM Ablation Analysis: DenseNet-121 (Figure 6.1)

```
========================================================================================
                      CBAM ABLATION STUDY: DENSENET-121
========================================================================================
METRIC                     BASELINE (NO CBAM)      HYBRID (+ CBAM)         DELTA (Δ)
----------------------------------------------------------------------------------------
Test Accuracy              99.24% (2,219/2,236)    98.57% (2,204/2,236)    -0.67%
Macro Sensitivity          99.27%                  98.66%                  -0.61%
Macro Specificity          99.61%                  99.29%                  -0.32%
Macro ROC-AUC              0.9998                  0.9997                  -0.0001
TB Sensitivity (Recall)    99.53% (631/634)        98.74% (626/634)        -0.79%
Pneumonia Recall           99.26% (668/673)        99.41% (669/673)        +0.15%
Total Misclassifications   17 Scans                32 Scans                +15 Scans
========================================================================================
```

---

## 6. Technical Verification & Reproducibility Instructions

To regenerate all evaluation figures:
```bash
# Navigate to repository root
cd "E:\Git All Repo\hybrid-tuberculosis-pneumonia-detection-xai"

# Generate evaluation figures
python src/generate_eval_figures.py
python src/generate_resnet50_figures.py
python src/generate_efficientnet_figures.py
python src/generate_densenet_baseline_figures.py
```
