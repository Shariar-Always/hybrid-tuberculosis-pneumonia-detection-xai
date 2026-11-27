# Clinical Benchmark & Experimental Evaluation Report: DenseNet-121 Baseline (No Attention)
## Multicenter Deep Learning Evaluation on 22,356 Digital Chest Radiographs

**Model Name:** Standard DenseNet-121 Baseline (Without Attention Mechanisms)  
**Experiment ID:** `densenet121_baseline_22k`  
**Dataset Scale:** 22,356 Scans (Train: 19,002 | Val: 1,118 | Test: 2,236)  
**Task:** Tri-Class Pulmonary Pathology Classification (NORMAL | TUBERCULOSIS | PNEUMONIA)  
**Target Hardware:** NVIDIA Tesla T4 GPU (Google Compute Engine)  
**Checkpoint Path (Local):** `models/best_tb_model_densenet121_baseline.pth` (29.5 MB)  
**Metrics Log (Local):** `models/metrics_densenet121_baseline.json`  
**History Log (Local):** `models/history_densenet121_baseline.json`  

---

## 1. Executive Performance Summary

The standard DenseNet-121 baseline architecture (operating without Channel or Spatial Attention Modules) was evaluated on the independent hold-out clinical test cohort of **2,236 digital chest radiographs**. DenseNet-121 leverages dense feature connectivity where each layer receives direct feature inputs from all preceding layers, promoting maximum feature reuse:

| Key Clinical Evaluation Metric | Empirical Test Result | Clinical Significance & Standard |
| :--- | :---: | :--- |
| **Overall Test Accuracy** | **99.24%** | 2,219 correctly classified scans out of 2,236 independent tests (only 17 misclassifications). |
| **Macro Sensitivity (Recall)** | **99.27%** | Outstanding pathology recall across Normal, Tuberculosis, and Pneumonia. |
| **Macro Specificity** | **99.61%** | Negligible false-positive rate ($<0.4\%$), preventing unwarranted hospital admissions. |
| **Macro ROC-AUC** | **0.9998** | Near-flawless discrimination across all operating thresholds. |
| **Tuberculosis Recall / F1-Score** | **0.9953 (99.5%) / 0.99** | 631 out of 634 active TB examinations correctly diagnosed. |
| **Pneumonia Recall / F1-Score** | **0.9926 (99.3%) / 0.99** | 668 out of 673 pneumonia cases correctly diagnosed; zero cross-confusion with TB. |
| **Normal Recall / F1-Score** | **0.9903 (99.0%) / 0.99** | 920 out of 929 healthy controls correctly identified. |
| **Model Footprint / Checkpoint Size** | **6.96M Parameters / 29.5 MB** | Highly compact convolutional topology suitable for hospital workstations. |

---

## 2. Epoch-by-Epoch Training & Validation Convergence Dynamics

Training was executed across 15 full epochs utilizing AdamW optimization ($\beta_1=0.9, \beta_2=0.999$, weight decay $=1 \times 10^{-2}$), initial learning rate of $1 \times 10^{-4}$, dynamic learning rate decay via `ReduceLROnPlateau` (patience $=3$, factor $=0.5$), and weighted Cross-Entropy Loss to balance class representation across Normal (9,285), Tuberculosis (7,481), and Pneumonia (5,590):

| Epoch | Elapsed Time (s) | Training Loss | Validation Loss | Validation Accuracy (%) | Validation Recall (%) | Validation ROC-AUC | Best Checkpoint Saved |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 395.1s | 0.2721 | 0.1990 | 93.29% | 92.54% | 0.9934 | **YES (Saved)** |
| **2** | 393.4s | 0.1491 | 0.1110 | 95.89% | 96.36% | 0.9971 | **YES (Saved)** |
| **3** | 396.7s | 0.1249 | 0.2124 | 94.01% | 93.66% | 0.9878 | No |
| **4** | 396.1s | 0.1025 | 0.0590 | 97.94% | 98.10% | 0.9987 | **YES (Saved)** |
| **5** | 392.2s | 0.0870 | 0.1788 | 94.28% | 93.63% | 0.9950 | No |
| **6** | 392.0s | 0.0780 | 0.0955 | 96.60% | 96.88% | 0.9982 | No |
| **7** | 391.5s | 0.0792 | 0.0643 | 97.67% | 97.56% | 0.9990 | No |
| **8** | 393.9s | 0.0442 | 0.0431 | 98.75% | 98.81% | 0.9993 | **YES (Saved)** |
| **9** | 397.6s | 0.0388 | 0.0666 | 97.58% | 97.45% | 0.9991 | No |
| **10** | 393.5s | 0.0366 | 0.0387 | 98.75% | 98.74% | 0.9994 | **YES (Saved)** |
| **11** | 404.6s | 0.0341 | 0.0534 | 98.03% | 98.24% | 0.9992 | No |
| **12** | 401.6s | 0.0292 | 0.0504 | 98.48% | 98.48% | 0.9992 | No |
| **13** | 403.0s | 0.0304 | 0.0537 | 98.48% | 98.40% | 0.9991 | No |
| **14** | 403.1s | 0.0177 | 0.0350 | 98.84% | 98.89% | 0.9994 | **YES (Saved - Optimal)** |
| **15** | 403.0s | 0.0157 | 0.0369 | 98.93% | 99.01% | 0.9990 | No |

### Analytical Insights from Training Dynamics:
1. **Convergence Trajectory:** Training loss exhibited a continuous, stable decrease from $0.2721$ in Epoch 1 down to $0.0157$ in Epoch 15 (a 17.3-fold reduction).
2. **Optimal Generalization Checkpoint:** The minimum validation loss of **$0.0350$** was reached at **Epoch 14** with a corresponding validation accuracy of **$98.84\%$** and sensitivity of **$98.89\%$**. These weights were preserved as `best_tb_model_densenet121_baseline.pth`.
3. **Computational Cost:** DenseNet-121 baseline required an average of $397.1$ seconds per epoch, totaling approximately 5,957 seconds (~99.3 minutes) for 15 complete passes over the 19,002 training radiographs.

---

## 3. Independent Hold-Out Test Set Evaluation (N = 2,236 Scans)

The evaluation was conducted on strictly unseen chest radiographs ($n = 2,236$) curated from the 8-center clinical benchmark:

```
============================================================
  Final Test Evaluation (densenet121 - Baseline No CBAM)
============================================================
Test Accuracy:             99.24%
Test Sensitivity (Recall): 99.27%
Test Specificity:          99.61%
Test ROC-AUC:              0.9998
============================================================
              precision    recall  f1-score   support

      Normal       0.99      0.99      0.99       929
Tuberculosis       0.99      1.00      0.99       634
   Pneumonia       1.00      0.99      0.99       673

    accuracy                           0.99      2236
   macro avg       0.99      0.99      0.99      2236
weighted avg       0.99      0.99      0.99      2236
```

### Detailed Class-Wise Performance Breakdown:
- **Normal Pulmonary Cohort ($n = 929$):**
  - Precision: **0.99** (920 / 927 predicted Normal)
  - Recall (Sensitivity): **0.99** (920 / 929 true Normal)
  - F1-Score: **0.99**
  - Clinical finding: 9 false positives (6 referred as suspected TB, 3 as Pneumonia).
- **Active Tuberculosis Cohort ($n = 634$):**
  - Precision: **0.99** (631 / 638 predicted TB)
  - Recall (Sensitivity): **0.9953 (~1.00)** (631 / 634 true TB)
  - F1-Score: **0.99**
  - Clinical finding: Only 3 false negatives out of 634 confirmed TB patients (all 3 classified as Normal; 0 misdiagnosed as Pneumonia).
- **Pneumonia Cohort ($n = 673$):**
  - Precision: **1.00** (668 / 671 predicted Pneumonia)
  - Recall (Sensitivity): **0.9926 (99.3%)** (668 / 673 true Pneumonia)
  - F1-Score: **0.99**
  - Clinical finding: Only 5 misclassifications (4 classified as Normal, 1 as TB).

---

## 4. Quantitative 3x3 Confusion Matrix Distribution

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

### Key Diagnostic Observations:
1. **Zero TB-to-Pneumonia Cross-Contamination:** Exactly 0 out of 634 confirmed tuberculosis patients were misclassified as pneumonia, preventing inappropriate antibiotic administration when antimycobacterial therapy is urgently indicated.
2. **Single Pneumonia-to-TB Case:** Only 1 out of 673 pneumonia patients was categorized as suspected TB, reflecting exceptional diagnostic specificity.
3. **Total Error Count:** Only 17 total classification errors occurred across 2,236 examinations (error rate of $0.76\%$).

---

## 5. CBAM Ablation Study: DenseNet-121 Baseline vs. DenseNet-121 + CBAM Hybrid

A critical objective of this benchmark is the rigorous ablation of attention mechanisms within identical backbone architectures:

| Diagnostic Performance Metric | DenseNet-121 Baseline (No Attention) | DenseNet-121 + CBAM (Hybrid) | Empirical Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **99.24%** (2,219/2,236) | 98.57% (2,204/2,236) | $-0.67\%$ |
| **Macro Sensitivity (Recall)** | **99.27%** | 98.66% | $-0.61\%$ |
| **Macro Specificity** | **99.61%** | 99.29% | $-0.32\%$ |
| **Macro ROC-AUC** | **0.9998** | 0.9997 | $-0.0001$ |
| **TB Recall (Sensitivity)** | **99.53%** (631/634) | 98.74% (626/634) | $-0.79\%$ |
| **Pneumonia Recall** | 99.26% (668/673) | **99.41%** (669/673) | $+0.15\%$ |
| **Total Test Misclassifications** | **17 Scans** | 32 Scans | $+15$ Scans |
| **Model Size / Parameter Count** | **29.5 MB / 6.96M** | 30.0 MB / 7.04M | $+0.5\text{ MB} / +0.08\text{M}$ |

### Academic Discussion of Ablation Dynamics:
- **Dense Connectivity & Feature Reuse:** DenseNet-121 already incorporates extensive feature concatenation across all layers within each dense block. This architectural inductive bias inherently mitigates vanishing gradients and preserves multi-scale spatial representations.
- **Attention Regularization vs. Empirical Fit:** On this particular stratified test split, the baseline DenseNet-121 achieved an extraordinary 99.24% accuracy. The addition of CBAM attention slightly modified decision margins, leading to 98.57% accuracy. However, as demonstrated in the Explainable AI (XAI) analysis (Category 5), the CBAM hybrid variant produces significantly tighter, more radiologically faithful gradient heatmaps (Grad-CAM and Grad-CAM++), strictly localizing apical cavitary lesions rather than relying on diffuse lung parenchymal features.

---

## 6. Generated Publication Figures

| Figure Identifier | Filename | Format | Description |
| :--- | :--- | :--- | :--- |
| **Figure 3.10** | `fig3_10_densenet121_baseline_loss_convergence.png` (.pdf) | 300 DPI / Vector | Epoch vs Training and Validation Cross-Entropy Loss |
| **Figure 3.11** | `fig3_11_densenet121_baseline_accuracy_progression.png` (.pdf) | 300 DPI / Vector | Epoch vs Validation Accuracy and Sensitivity |
| **Figure 3.12** | `fig3_12_densenet121_baseline_composite_dashboard.png` (.pdf) | 300 DPI / Vector | Dual-panel composite publication training dashboard |
| **Figure 4.2E** | `fig4_2e_densenet121_baseline_confusion_matrix.png` (.pdf) | 300 DPI / Vector | 3x3 holdout test set confusion matrix with patient counts |
| **Figure 6.1** | `fig6_1_cbam_ablation_densenet121.png` (.pdf) | 300 DPI / Vector | Direct CBAM attention ablation comparison bar chart |

---

## 7. Manuscript LaTeX Table Snippet

```latex
\begin{table*}[t]
\centering
\caption{Empirical Diagnostic Performance of DenseNet-121 Baseline (No Attention) on 2,236 Multi-Center Hold-Out Radiographs.}
\label{tab:densenet121_baseline_benchmark}
\begin{tabular}{lcccccc}
\hline
\textbf{Pathology Class} & \textbf{Support ($N$)} & \textbf{Precision} & \textbf{Recall (Sensitivity)} & \textbf{Specificity} & \textbf{F1-Score} & \textbf{ROC-AUC} \\
\hline
Normal & 929 & 0.99 & 0.9903 & 0.9931 & 0.99 & 0.9998 \\
Tuberculosis & 634 & 0.99 & 0.9953 & 0.9956 & 0.99 & 0.9999 \\
Pneumonia & 673 & 1.00 & 0.9926 & 0.9981 & 0.99 & 0.9998 \\
\hline
\textbf{Macro Average / Overall} & \textbf{2,236} & \textbf{0.99} & \textbf{99.27\%} & \textbf{99.61\%} & \textbf{0.99} & \textbf{0.9998} \\
\hline
\end{tabular}
\end{table*}
```
