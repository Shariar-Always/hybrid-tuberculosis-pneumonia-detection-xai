# Agent Directives & Project Rules (Agent/agent.md)
**Project:** Hybrid Deep Learning Framework for Tuberculosis & Pneumonia Detection with XAI  
**Workspace:** `E:\Git All Repo\hybrid-tuberculosis-pneumonia-detection-xai`

---

## ⛔ CRITICAL RULE 1: GIT PUSH RESTRICTION (STRICT USER PERMISSION REQUIRED)
> **🚫 MANDATORY DIRECTIVE:**  
> **Under NO circumstances should the agent execute a `git push` to GitLab (`git push gitlab`), GitHub (`git push origin`), or Hugging Face (`git push hf`) without explicit user permission or direct command.**

### Protocols:
1. **Local Changes Only:** All code modifications, dataset indexing, and documentation must remain strictly local on the machine unless the user explicitly commands: *"Push to GitLab"*, *"Push now"*, etc.
2. **Commit vs. Push:** Local commits (`git commit`) may be prepared locally when necessary, but **NEVER run `git push`** without direct user approval.
3. **Always Ask / Wait:** When changes are ready, notify the user that local modifications are complete and wait for their explicit go-ahead before pushing.

---

## 📌 CORE OPERATIONAL DIRECTIVES

### 1. Language & Documentation Standards:
- All documentation files, markdown tables, reports, code comments, and metadata descriptions must be written in **100% formal academic English**.
- Strictly avoid fabricating metrics or experimental numbers; maintain authentic, verifiable results.

### 2. Dataset Management & Provenance Integrity:
- **Clean Dataset Location:** `D:\TB N\dataset_clean`
  - Total Verified Images: **22,356** (NORMAL: 9,285, TUBERCULOSIS: 7,481, PNEUMONIA: 5,590).
  - Stratification: 85% Train (19,002), 5% Val (1,118), 10% Test (2,236).
- **Metadata Files:**
  - `D:\TB N\dataset_clean\metadata.csv` (22,356 rows, 18 clinical columns)
  - `data/metadata.csv` (22,356 rows, active project index)
  - `data/metadata_22k_clinical.csv` (22,356 rows, clinical provenance)
  - `data/metadata_phase1_13k.csv` (13,959 rows, Phase 1 benchmark baseline)
- **Clinical Traceability:** Every single image row must preserve its clinical hospital origin across all 8 partner institutions (Guangzhou, TBX11K, NIH RSNA, NIAID, Shenzhen, Belarus, Montgomery, Qatar/Dhaka, Mendeley Pakistan).

### 3. Model Training & XAI Architecture:
- Framework: PyTorch.
- 6 Models to train and benchmark:
  - 3 Hybrid Architectures (with CBAM Attention): DenseNet-121, ResNet-50, EfficientNet-B0.
  - 3 Baseline / Normal Architectures (without CBAM): DenseNet-121, ResNet-50, EfficientNet-B0.
- Explainable AI Suite: Grad-CAM, Grad-CAM++, Guided Backpropagation, Saliency Maps.
- Evaluation Metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, Confusion Matrix.
