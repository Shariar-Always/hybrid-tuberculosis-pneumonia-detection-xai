# Training Convergence Dynamics & Learning Trajectory Figures
## Academic Methodology, Statistical Analysis & Thesis Documentation

**Directory:** `reports/figures/training/`  
**Model Architectures:** 
- Hybrid DenseNet-121 + CBAM (`fig3_1`, `fig3_2`, `fig3_3`)
- Hybrid ResNet-50 + CBAM (`fig3_4`, `fig3_5`, `fig3_6`)  
**Dataset Scale:** 22,356 Digital Chest Radiographs (Training: 19,002 | Validation: 1,118 | Test: 2,236)  
**Training Hardware:** NVIDIA Tesla T4 GPU (Google Compute Engine)  
**Graphics Standards:** 300+ DPI Raster PNG and Vector PDF (IEEE / Elsevier / Nature Publication Grade)  
**Associated Generator Scripts:** `src/generate_training_figures.py` and `src/generate_resnet50_figures.py`  

---

## 1. Executive Summary

This documentation details the empirical training and validation convergence dynamics of the hybrid deep learning architectures trained on the expanded multicenter 22,356 digital chest radiograph cohort.

In computational medical imaging manuscripts, learning curve figures serve as primary empirical proof for:
1. **Optimization Stability:** Demonstrating smooth, asymptotic cross-entropy loss minimization without gradient explosion or severe oscillations.
2. **Generalization Integrity & Regularization:** Verifying that validation metrics closely track training trajectories without divergent overfitting.
3. **Attention Integration Efficacy:** Documenting how Channel and Spatial Attention Modules accelerate convergence velocity towards low loss minima ($<0.04$).

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
| `TRAINING_CONVERGENCE_EXPLANATION.md` | Markdown | Academic Documentation | Comprehensive methodology, analytical breakdown, and LaTeX snippets |

---

## 3. ResNet-50 + CBAM Convergence Trajectory (Figures 3.4, 3.5, 3.6)

### Visual Breakdown:
- **Figure 3.4 (Loss Convergence):** 
  - Training loss decreased monotonically from $0.3121$ in Epoch 1 to $0.0192$ in Epoch 15 (a 16.2-fold minimization).
  - Validation loss reached its global minimum of **$0.0334$ at Epoch 12**.
  - A transient validation spike at Epoch 7 ($0.7204$) was smoothly stabilized by the `ReduceLROnPlateau` scheduler, dropping to $0.0518$ by Epoch 8 and maintaining steady sub-0.05 validation loss thereafter.
- **Figure 3.5 (Accuracy & Recall Progression):**
  - Validation accuracy started strong at $96.60\%$ in Epoch 1 and peaked at **$98.93\%$ in Epoch 13**.
  - Validation sensitivity (macro recall) attained **$98.93\%$ in Epoch 13**.
- **Figure 3.6 (Composite Dashboard):**
  - Panel A details loss progression across saved checkpoint epochs (1, 2, 4, 5, 6, 8, 12).
  - Panel B details simultaneous tracking of Validation Accuracy ($98.93\%$) and Validation ROC-AUC ($0.9995$).

### Ready-to-Use Manuscript Text (For Chapter 4):
> *"The optimization trajectory of the Hybrid ResNet-50 + CBAM model demonstrated strong convergence characteristics on the 22,356 chest radiograph dataset (Figures 3.4-3.6). Initialized with pre-trained weights, the training loss decayed smoothly from 0.3121 to 0.0192. Following a transient validation perturbation at Epoch 7, the adaptive learning rate scheduler halved the optimization step size, prompting immediate recovery to 98.21% accuracy in Epoch 8 and an optimum validation loss of 0.0334 at Epoch 12. The network achieved a peak validation accuracy of 98.93% and a validation ROC-AUC of 0.9995, corroborating the stability of residual learning enhanced by dual-domain attention modules."*

### Copy-Paste LaTeX Figure Code:
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{reports/figures/training/fig3_6_resnet50_composite_training_dashboard.pdf}
\caption{\textbf{Composite Training Dynamics and Optimization Dashboard for Hybrid ResNet-50 + CBAM.} 
\textbf{(A)} Cross-entropy loss minimization across 15 epochs with markers indicating model checkpoints saved to disk. 
\textbf{(B)} Trajectory of validation accuracy (peaking at $98.93\%$) and validation ROC-AUC ($0.9995$), demonstrating robust convergence on the 22,356 multi-center clinical dataset.}
\label{fig:resnet50_training_dashboard}
\end{figure*}
```

---

## 4. DenseNet-121 + CBAM Convergence Trajectory (Figures 3.1, 3.2, 3.3)

### Visual Breakdown:
- **Figure 3.1 (Loss Convergence):** 
  - Training loss decreased monotonically from $0.2998$ to $0.0277$.
  - Validation loss reached its minimum of **$0.0386$ in Epoch 15** (with interim troughs of $0.0423$ at Epoch 12 and $0.0491$ at Epoch 8).
- **Figure 3.2 (Accuracy & Recall Progression):**
  - Validation accuracy started at $96.06\%$ in Epoch 1, peaking at **$98.84\%$ in Epoch 8**.
  - Validation sensitivity tracked identically, attaining **$98.86\%$ in Epoch 8**.
- **Figure 3.3 (Composite Dashboard):**
  - Double-column journal figure combining cross-entropy loss and accuracy/ROC-AUC trajectory.

---

## 5. Comparative Convergence Analysis: DenseNet-121 vs. ResNet-50

| Metric | DenseNet-121 + CBAM | ResNet-50 + CBAM | Comparison & Observation |
| :--- | :---: | :---: | :--- |
| **Initial Epoch 1 Val Accuracy** | 96.06% | **96.60%** | Both architectures achieve immediate $>96\%$ accuracy. |
| **Minimum Validation Loss** | 0.0386 (Epoch 15) | **0.0334 (Epoch 12)** | ResNet-50 reaches a lower loss minimum earlier. |
| **Peak Validation Accuracy** | 98.84% (Epoch 8) | **98.93% (Epoch 13)** | ResNet-50 achieves slightly higher validation fidelity. |
| **Peak Validation ROC-AUC** | 0.9994 (Epoch 12) | **0.9995 (Epoch 12)** | Both models exhibit near-perfect area under the curve. |
| **Final Training Loss** | 0.0277 | **0.0192** | ResNet-50 achieves tighter training error convergence. |
| **Average Epoch Time (T4 GPU)** | 393.4s | **388.9s** | Both models train in approximately 6.5 minutes per epoch. |

---

## 6. Technical Verification & Reproducibility Instructions

To regenerate all training convergence figures:
```bash
# Navigate to repository root
cd "E:\Git All Repo\hybrid-tuberculosis-pneumonia-detection-xai"

# Generate DenseNet-121 training figures
python src/generate_training_figures.py

# Generate ResNet-50 training figures
python src/generate_resnet50_figures.py
```
All figures are exported at 300+ DPI in both raster PNG and vector PDF formats.
