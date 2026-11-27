# Training Convergence Dynamics & Learning Trajectory Figures
## Academic Methodology, Statistical Analysis & Thesis Documentation

**Directory:** `reports/figures/training/`  
**Model Architectures:** 
- Hybrid DenseNet-121 + CBAM (`fig3_1`, `fig3_2`, `fig3_3`)
- Hybrid ResNet-50 + CBAM (`fig3_4`, `fig3_5`, `fig3_6`)  
- Hybrid EfficientNet-B0 + CBAM (`fig3_7`, `fig3_8`, `fig3_9`)  
**Dataset Scale:** 22,356 Digital Chest Radiographs (Training: 19,002 | Validation: 1,118 | Test: 2,236)  
**Training Hardware:** NVIDIA Tesla T4 GPU (Google Compute Engine)  
**Graphics Standards:** 300+ DPI Raster PNG and Vector PDF (IEEE / Elsevier / Nature Publication Grade)  
**Associated Generator Scripts:** `src/generate_training_figures.py`, `src/generate_resnet50_figures.py`, and `src/generate_efficientnet_figures.py`  

---

## 1. Executive Summary

This documentation details the empirical training and validation convergence dynamics of the three proposed hybrid deep learning architectures trained on the expanded multicenter 22,356 digital chest radiograph cohort.

In computational medical imaging manuscripts, learning curve figures serve as primary empirical proof for:
1. **Optimization Stability:** Demonstrating smooth, asymptotic cross-entropy loss minimization without gradient explosion or severe oscillations.
2. **Generalization Integrity & Regularization:** Verifying that validation metrics closely track training trajectories without divergent overfitting.
3. **Attention Integration Efficacy:** Documenting how Channel and Spatial Attention Modules accelerate convergence velocity towards low loss minima ($<0.045$).

---

## 2. Directory Contents & Generated Assets

| Filename | Format | Dimensions / DPI | Description & Purpose |
| :--- | :--- | :--- | :--- |
| `fig3_1_densenet121_loss_convergence.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.1: Epoch vs Cross-Entropy Loss Convergence (DenseNet-121 + CBAM) |
| `fig3_1_densenet121_loss_convergence.pdf` | Vector PDF | Scale-Independent Vector | Vector format for DenseNet-121 loss trajectory |
| `fig3_2_densenet121_accuracy_progression.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.2: Epoch vs Validation Accuracy & Sensitivity (DenseNet-121 + CBAM) |
| `fig3_2_densenet121_accuracy_progression.pdf` | Vector PDF | Scale-Independent Vector | Vector format for DenseNet-121 accuracy progression |
| `fig3_3_densenet121_composite_training_dashboard.png` | Raster PNG | $4650 \times 1860$ (300 DPI) | Figure 3.3: Two-Panel Composite Dashboard (DenseNet-121 + CBAM) |
| `fig3_3_densenet121_composite_training_dashboard.pdf` | Vector PDF | Scale-Independent Vector | Dual-panel vector composite for journal publication |
| `fig3_4_resnet50_loss_convergence.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.4: Epoch vs Cross-Entropy Loss Convergence (ResNet-50 + CBAM) |
| `fig3_4_resnet50_loss_convergence.pdf` | Vector PDF | Scale-Independent Vector | Vector format for ResNet-50 loss trajectory |
| `fig3_5_resnet50_accuracy_progression.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.5: Epoch vs Validation Accuracy & Sensitivity (ResNet-50 + CBAM) |
| `fig3_5_resnet50_accuracy_progression.pdf` | Vector PDF | Scale-Independent Vector | Vector format for ResNet-50 accuracy progression |
| `fig3_6_resnet50_composite_training_dashboard.png` | Raster PNG | $4650 \times 1860$ (300 DPI) | Figure 3.6: Two-Panel Composite Dashboard (ResNet-50 + CBAM) |
| `fig3_6_resnet50_composite_training_dashboard.pdf` | Vector PDF | Scale-Independent Vector | Dual-panel vector composite for journal publication |
| `fig3_7_efficientnet_loss_convergence.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.7: Epoch vs Cross-Entropy Loss Convergence (EfficientNet-B0 + CBAM) |
| `fig3_7_efficientnet_loss_convergence.pdf` | Vector PDF | Scale-Independent Vector | Vector format for EfficientNet-B0 loss trajectory |
| `fig3_8_efficientnet_accuracy_progression.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.8: Epoch vs Validation Accuracy & Sensitivity (EfficientNet-B0 + CBAM) |
| `fig3_8_efficientnet_accuracy_progression.pdf` | Vector PDF | Scale-Independent Vector | Vector format for EfficientNet-B0 accuracy progression |
| `fig3_9_efficientnet_composite_training_dashboard.png` | Raster PNG | $4650 \times 1860$ (300 DPI) | Figure 3.9: Two-Panel Composite Dashboard (EfficientNet-B0 + CBAM) |
| `fig3_9_efficientnet_composite_training_dashboard.pdf` | Vector PDF | Scale-Independent Vector | Dual-panel vector composite for journal publication |
| `TRAINING_CONVERGENCE_EXPLANATION.md` | Markdown | Academic Documentation | Comprehensive methodology, analytical breakdown, and LaTeX snippets |

---

## 3. EfficientNet-B0 + CBAM Convergence Trajectory (Figures 3.7, 3.8, 3.9)

### Visual Breakdown:
- **Figure 3.7 (Loss Convergence):** 
  - Training loss decreased monotonically from $0.2829$ in Epoch 1 to $0.0167$ in Epoch 15 (a 16.9-fold minimization).
  - Validation loss reached its global minimum of **$0.0420$ at Epoch 12**.
- **Figure 3.8 (Accuracy & Recall Progression):**
  - Validation accuracy reached an immediate $96.78\%$ in Epoch 1 and peaked at **$98.93\%$ in Epoch 10 & 12**.
  - Validation sensitivity tracked identically, reaching **$98.90\%$ in Epoch 12**.
- **Figure 3.9 (Composite Dashboard):**
  - Panel A details loss progression across saved checkpoint epochs (1, 2, 3, 4, 5, 8, 12).
  - Panel B details simultaneous tracking of Validation Accuracy ($98.93\%$) and Validation ROC-AUC ($0.9991$).

---

## 4. ResNet-50 + CBAM Convergence Trajectory (Figures 3.4, 3.5, 3.6)

### Visual Breakdown:
- **Figure 3.4 (Loss Convergence):** 
  - Training loss decreased from $0.3121$ to $0.0192$.
  - Validation loss reached its global minimum of **$0.0334$ at Epoch 12**.
- **Figure 3.5 (Accuracy & Recall Progression):**
  - Validation accuracy reached **$98.93\%$ in Epoch 13** with macro recall of **$98.93\%$**.
- **Figure 3.6 (Composite Dashboard):**
  - Two-panel journal composite illustrating loss decay and accuracy/AUC progression.

---

## 5. DenseNet-121 + CBAM Convergence Trajectory (Figures 3.1, 3.2, 3.3)

### Visual Breakdown:
- **Figure 3.1 (Loss Convergence):** 
  - Training loss decreased from $0.2998$ to $0.0277$.
  - Validation loss reached **$0.0386$ in Epoch 15**.
- **Figure 3.2 (Accuracy & Recall Progression):**
  - Validation accuracy peaked at **$98.84\%$ at Epoch 8** ($98.86\%$ recall).

---

## 6. Comprehensive Tri-Model Convergence Comparison

| Metric | DenseNet-121 + CBAM | ResNet-50 + CBAM | EfficientNet-B0 + CBAM | Analytical Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Initial Epoch 1 Val Accuracy** | 96.06% | 96.60% | **96.78%** | All 3 models achieve immediate $>96\%$ accuracy. |
| **Minimum Validation Loss** | 0.0386 (Epoch 15) | **0.0334 (Epoch 12)** | 0.0420 (Epoch 12) | ResNet-50 attains lowest loss minimum. |
| **Peak Validation Accuracy** | 98.84% (Epoch 8) | **98.93% (Epoch 13)** | **98.93% (Epoch 10/12)** | Both ResNet and EfficientNet attain 98.93%. |
| **Peak Validation ROC-AUC** | 0.9994 | **0.9995** | 0.9991 | All architectures exceed 0.999 AUC. |
| **Final Training Loss** | 0.0277 | 0.0192 | **0.0167** | EfficientNet achieves lowest training error. |
| **Average Epoch Time (T4 GPU)** | 393.4s | 388.9s | **376.4s** | **EfficientNet-B0 is the fastest model (6.2 min/epoch).** |

---

## 7. Technical Verification & Reproducibility Instructions

To regenerate all training convergence figures:
```bash
# Navigate to repository root
cd "E:\Git All Repo\hybrid-tuberculosis-pneumonia-detection-xai"

# Generate figures for all three architectures
python src/generate_training_figures.py
python src/generate_resnet50_figures.py
python src/generate_efficientnet_figures.py
```
All figures are exported at 300+ DPI in both raster PNG and vector PDF formats.
