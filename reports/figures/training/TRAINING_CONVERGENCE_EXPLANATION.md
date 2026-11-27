# Training Convergence Dynamics & Learning Trajectory Figures
## Academic Methodology, Statistical Analysis & Thesis Documentation

**Directory:** `reports/figures/training/`  
**Model Architecture:** Hybrid DenseNet-121 with Convolutional Block Attention Modules (CBAM)  
**Dataset Scale:** 22,356 Digital Chest Radiographs (Training: 19,002 | Validation: 1,118 | Test: 2,236)  
**Training Hardware:** NVIDIA Tesla T4 GPU (Google Compute Engine)  
**Graphics Standards:** 300+ DPI Raster PNG and Vector PDF (IEEE / Elsevier / Nature Publication Grade)  
**Associated Generator Script:** `src/generate_training_figures.py`  

---

## 1. Executive Summary

This documentation details the empirical training and validation convergence dynamics of the champion deep learning model, **Hybrid DenseNet-121 + CBAM**, trained on the full multicenter 22,356 digital chest radiograph cohort.

In computational medical imaging manuscripts, learning curve figures serve as primary empirical proof for:
1. **Optimization Stability:** Demonstrating smooth, asymptotic cross-entropy loss minimization without gradient explosion or severe oscillations.
2. **Generalization Integrity & Regularization:** Verifying that validation metrics closely track training trajectories without divergent overfitting.
3. **Attention Integration Efficacy:** Documenting how Channel and Spatial Attention Modules accelerate convergence velocity towards low loss minima ($<0.04$).

---

## 2. Directory Contents & Generated Assets

| Filename | Format | Dimensions / DPI | Description & Purpose |
| :--- | :--- | :--- | :--- |
| `fig3_1_densenet121_loss_convergence.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.1: Epoch vs Cross-Entropy Loss Convergence (Train Loss vs Val Loss) |
| `fig3_1_densenet121_loss_convergence.pdf` | Vector PDF | Scale-Independent Vector | Vector format suitable for single-column thesis and journal submission |
| `fig3_2_densenet121_accuracy_progression.png` | Raster PNG | $2550 \times 1650$ (300 DPI) | Figure 3.2: Epoch vs Validation Accuracy & Sensitivity Progression |
| `fig3_2_densenet121_accuracy_progression.pdf` | Vector PDF | Scale-Independent Vector | Vector format for high-resolution manuscript printing |
| `fig3_3_densenet121_composite_training_dashboard.png` | Raster PNG | $4650 \times 1860$ (300 DPI) | Figure 3.3: Two-Panel Composite Training Dashboard (Loss + Accuracy/AUC) |
| `fig3_3_densenet121_composite_training_dashboard.pdf` | Vector PDF | Scale-Independent Vector | Two-column composite figure matching Nature/IEEE format |
| `TRAINING_CONVERGENCE_EXPLANATION.md` | Markdown | Academic Documentation | Detailed thesis methodology write-up, data tables, and LaTeX snippets |

---

## 3. Detailed Figure Specifications & Thesis Text

### Figure 3.1: Training & Validation Loss Convergence

#### A. Visual Breakdown
Directly mapping demo reference asset `image copy.png`:
- **Training Loss (Solid Red Line with circular markers):** Shows monotonic decay from $0.2998$ in Epoch 1 to $0.0277$ in Epoch 15 (a $>10\times$ minimization).
- **Validation Loss (Dashed Blue Line with square markers):** Displays rapid initial decline to $0.1175$ at Epoch 1, with persistent reduction reaching its absolute minimum at **$0.0386$ in Epoch 15** (annotated with a green highlighted callout).

#### B. Ready-to-Use Manuscript Text (For Thesis Chapter 4: Training Dynamics)
> *"The optimization trajectory of the proposed Hybrid DenseNet-121 + CBAM architecture demonstrated exceptional numerical stability over 15 training epochs (Figure 3.1). Initialized with ImageNet-1k pre-trained weights and fine-tuned using the AdamW optimizer ($lr = 1 \times 10^{-4}, \text{weight decay} = 1 \times 10^{-4}$), the training cross-entropy loss decreased monotonically from 0.2998 to 0.0277. The validation loss mirrored this trajectory, falling from 0.1175 in the initial epoch to a global minimum of 0.0386 at Epoch 15. The absence of divergent validation loss spikes confirms the regularization potency of CBAM spatial attention and the dropout layer ($p=0.3$) prior to the classification head, successfully mitigating overfitting on the 19,002 training radiographs."*

#### C. Copy-Paste LaTeX Figure Code
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{reports/figures/training/fig3_1_densenet121_loss_convergence.pdf}
\caption{\textbf{Cross-Entropy Loss Convergence Trajectory for Hybrid DenseNet-121 + CBAM.} 
Monotonic reduction of training loss ($0.2998 \to 0.0277$) and validation loss reaching a global minimum of $0.0386$ at Epoch 15, confirming optimization stability without overfitting across 22,356 clinical radiographs.}
\label{fig:loss_convergence}
\end{figure}
```

---

### Figure 3.2: Epoch vs Validation Accuracy & Sensitivity Progression

#### A. Visual Breakdown
Directly mapping demo reference asset `image copy 10.png`:
- **Validation Accuracy (Diamond Green Line):** Highlights immediate high performance starting at $96.06\%$ in Epoch 1, peaking at **$98.84\%$ at Epoch 8**, and concluding stably at $98.39\%$ at Epoch 15.
- **Validation Sensitivity / Recall (Triangle Purple Line):** Follows an identical high-sensitivity trajectory, attaining **$98.86\%$ at Epoch 8**, vital for pulmonary infection screening.

#### B. Ready-to-Use Manuscript Text
> *"Diagnostic accuracy and sensitivity progressions across training epochs are illustrated in Figure 3.2. Owing to the feature reuse characteristics of dense blocks and the dual-attention mechanism, the network surpassed 96.0% validation accuracy within the first epoch (96.06% accuracy, 96.09% recall). Performance peaked at Epoch 8 with 98.84% accuracy and 98.86% clinical sensitivity. The close tracking between overall accuracy and sensitivity substantiates balanced multi-class learning, ensuring that the model does not disproportionately favor the majority Normal class at the expense of Tuberculosis or Pneumonia detection."*

#### C. Copy-Paste LaTeX Figure Code
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{reports/figures/training/fig3_2_densenet121_accuracy_progression.pdf}
\caption{\textbf{Epoch vs. Validation Accuracy and Sensitivity Progression.} 
Diagnostic accuracy and recall trajectories across 15 epochs, achieving peak performance at Epoch 8 ($98.84\%$ accuracy, $98.86\%$ sensitivity) and maintaining superior clinical fidelity throughout training.}
\label{fig:accuracy_progression}
\end{figure}
```

---

### Figure 3.3: Two-Panel Composite Training Dashboard

#### A. Visual Breakdown
Directly combining `image copy.png` and `image copy 10.png` into a unified double-column journal panel:
- **Panel A (Left):** Loss minimization trajectories with vertical dotted lines indicating saved model checkpoints (Epochs 1, 3, 6, 8, 12, 15).
- **Panel B (Right):** Dual-axis progression showing Validation Accuracy (%) on the left y-axis and Validation ROC-AUC (reaching $0.9994$) on the right y-axis.

#### B. Copy-Paste LaTeX Figure Code (For IEEE Two-Column Papers)
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{reports/figures/training/fig3_3_densenet121_composite_training_dashboard.pdf}
\caption{\textbf{Composite Training Dynamics and Optimization Dashboard for Hybrid DenseNet-121 + CBAM.} 
\textbf{(A)} Cross-entropy loss minimization across 15 epochs with vertical markers indicating improved model checkpoints saved to disk. 
\textbf{(B)} Simultaneous progression of validation accuracy (left axis, peaking at $98.84\%$) and validation ROC-AUC (right axis, attaining $>0.999$), demonstrating rapid asymptotic convergence on the 22,356 multi-center dataset.}
\label{fig:composite_training_dashboard}
\end{figure*}
```

---

## 4. Technical Verification & Reproducibility Instructions

To regenerate these figures or integrate them with future architecture runs:
```bash
# Navigate to repository root
cd "E:\Git All Repo\hybrid-tuberculosis-pneumonia-detection-xai"

# Execute dedicated training figure generator
python src/generate_training_figures.py
```
All figures are exported at 300+ DPI in both raster PNG and vector PDF formats.
