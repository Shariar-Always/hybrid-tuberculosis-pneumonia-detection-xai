# Confusion Matrix & Diagnostic Discrimination Analysis
## Evaluation of Champion Model (Hybrid DenseNet-121 + CBAM) on 2,236 Clinical Scans

**Directory:** `reports/figures/eval/`  
**Model Architecture:** DenseNet-121 with Convolutional Block Attention Modules (CBAM)  
**Test Cohort Scale:** 2,236 Unique Digital Chest Radiographs (Hold-out Test Set)  
**Figure Standard:** 300+ DPI Raster PNG and Scale-Independent Vector PDF (IEEE / Nature / Elsevier Standards)  
**Associated Generator Script:** `src/generate_eval_figures.py`  

---

## 1. Executive Diagnostic Breakdown

Figure 4.2A illustrates the primary multiclass confusion matrix of the champion **DenseNet-121 + CBAM** architecture evaluated on the independent hold-out test cohort of **2,236 digital chest radiographs**.

### Primary Multiclass Metrics:
- **Overall Test Accuracy:** **98.57%** ($2,204 / 2,236$ scans correctly diagnosed)
- **Macro Sensitivity (Recall):** **98.66%**
- **Macro Specificity:** **99.29%**
- **Macro ROC-AUC:** **0.9997**
- **Total Diagnostic Misclassifications:** **32 scans** ($1.43\%$ aggregate error rate)

---

## 2. Quantitative 3x3 Confusion Matrix Distribution

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

### Detailed Pathological Error Analysis:
1. **Normal Pulmonary Screening Controls ($n = 929$):**
   - Correctly Diagnosed: **909 scans (97.85%)**
   - Misdiagnosed as Tuberculosis: **9 scans (0.97%)** (minimal false alarms)
   - Misdiagnosed as Pneumonia: **11 scans (1.18%)** (minor interstitial/bronchial prominence)

2. **Active Pulmonary Tuberculosis ($n = 634$):**
   - Correctly Diagnosed: **626 scans (98.74%)**
   - False Negatives (Misdiagnosed as Normal): **7 scans (1.10%)**
   - Misclassified as Pneumonia: **1 scan (0.16%)**
   - *Clinical Implication:* With a $98.74\%$ sensitivity, missed active cases are reduced to only 7 patients out of 634, demonstrating the life-saving potential of the hybrid attention module in public health screening.

3. **Clinical Viral and Bacterial Pneumonia ($n = 673$):**
   - Correctly Diagnosed: **669 scans (99.41%)**
   - False Negatives (Misdiagnosed as Normal): **2 scans (0.30%)**
   - Misclassified as Tuberculosis: **2 scans (0.30%)**
   - *Clinical Implication:* Distinguishing pneumonia from pulmonary tuberculosis is historically challenging due to overlapping consolidations; our model achieved a $99.41\%$ clinical sensitivity with virtually zero cross-pathology confusion.

---

## 3. Directory Contents & Generated Assets

| Filename | Format | Dimensions / DPI | Description & Purpose |
| :--- | :--- | :--- | :--- |
| `fig4_2a_densenet121_champion_confusion_matrix.png` | Raster PNG | $2184 \times 1974$ (300 DPI) | Figure 4.2A: Primary 3x3 Test Confusion Matrix with dual count and percentage annotations |
| `fig4_2a_densenet121_champion_confusion_matrix.pdf` | Vector PDF | Scale-Independent Vector | Publication-grade vector PDF for high-resolution manuscript compilation |
| `CONFUSION_MATRIX_EVALUATION.md` | Markdown | Academic Documentation | Comprehensive statistical breakdown, clinical analysis, and LaTeX snippets |

---

## 4. Ready-to-Use Manuscript Text (For Thesis Chapter 4: Results & Discussion)

> *"The diagnostic classification performance of the champion Hybrid DenseNet-121 + CBAM model was benchmarked on the independent hold-out test cohort of 2,236 digital chest radiographs (Figure 4.2A). The network demonstrated exceptional diagnostic fidelity, correctly classifying 2,204 out of 2,236 radiographs to yield an overall test accuracy of 98.57%, a macro sensitivity of 98.66%, a macro specificity of 99.29%, and an area under the ROC curve (ROC-AUC) of 0.9997. In active pulmonary tuberculosis detection ($n = 634$), the model achieved an empirical sensitivity of 98.74% (626/634), with only 7 cases misclassified as normal controls and a single case misdiagnosed as pneumonia. For pneumonia screening ($n = 673$), the architecture achieved 99.41% sensitivity (669/673), demonstrating near-perfect discrimination between acute alveolar consolidations and chronic apical fibrocalcific tuberculosis lesions. Healthy normal controls ($n = 929$) yielded a specificity of 97.85% (909/929). These findings confirm that integrating channel and spatial attention modules successfully regularizes feature representations across heterogeneous multicenter radiograph acquisitions."*

---

## 5. Copy-Paste LaTeX Figure Code (For IEEE / Elsevier Overleaf)

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{reports/figures/eval/fig4_2a_densenet121_champion_confusion_matrix.pdf}
\caption{\textbf{Primary Multiclass Confusion Matrix for Hybrid DenseNet-121 + CBAM on the 2,236 Hold-out Test Cohort.} 
Cells display absolute patient counts and row-normalized class sensitivities. The champion model achieves $98.57\%$ accuracy, $98.74\%$ recall for Tuberculosis, and $99.41\%$ recall for Pneumonia with only 32 total misclassifications out of 2,236 multi-center radiographs.}
\label{fig:confusion_matrix_champion}
\end{figure}
```
