<h1 align="center">🫁 Hybrid Explainable AI Framework for Automatic Tuberculosis & Pneumonia Detection from Chest X-Ray Images</h1>

<p align="center">
  <img src="asset/image/Header_Picture.png" alt="Hybrid TB Detection XAI Cover Page" width="100%">
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/ReportLab-PDF%20Report-success?style=for-the-badge">
</p>

<p align="center">
  <a href="https://tuberculosis-pneumonia-detection-xai.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-Active-green?style=flat-square&logo=streamlit" alt="Streamlit App"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <img src="https://img.shields.io/github/repo-size/Shariar-Ahamed/hybrid-tuberculosis-pneumonia-detection-xai">
  <img src="https://img.shields.io/github/last-commit/Shariar-Ahamed/hybrid-tuberculosis-pneumonia-detection-xai">
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/Shariar-Ahamed/hybrid-tuberculosis-pneumonia-detection-xai?style=social">
  <img src="https://img.shields.io/github/forks/Shariar-Ahamed/hybrid-tuberculosis-pneumonia-detection-xai?style=social">
</p>

<p align="center">
  <a href="https://wakatime.com/badge/user/c7433bc5-6f12-4c97-baea-430790fa608c/project/941af381-3b72-4873-b4c0-4409eae76717"><img src="https://wakatime.com/badge/user/c7433bc5-6f12-4c97-baea-430790fa608c/project/941af381-3b72-4873-b4c0-4409eae76717.svg" alt="wakatime"></a>
</p>

---

An end-to-end, clinically validated, deep learning framework for the automated multi-class classification of **Normal**, **Tuberculosis (TB)**, and **Pneumonia** from Chest X-Ray (CXR) images. The system integrates pre-trained Convolutional Neural Network (CNN) backbones with a **Convolutional Block Attention Module (CBAM)** (combining sequential Channel and Spatial Attention) and provides transparent clinical interpretability through **four Explainable AI (XAI)** visualization algorithms: **Grad-CAM**, **Grad-CAM++**, **Guided Backpropagation**, and **Saliency Maps**.

The project also includes an **Out-of-Distribution (OOD) CXR input validation engine**, an interactive **Streamlit clinical dashboard**, and an **in-memory automated PDF diagnostic report generator** with timestamps and doctor sign-off blocks.

---

## 📌 Live Demo & Deployment

- 🌐 **Live Web Application:** [https://tuberculosis-pneumonia-detection-xai.streamlit.app/](https://tuberculosis-pneumonia-detection-xai.streamlit.app/)
- 📄 **Dataset Metadata:** [`data/metadata.csv`](data/metadata.csv)
- 🧪 **Model Checkpoints:** [`models/`](models/)
- 🔬 **Thesis & Academic Documentation:** [`thesis/`](thesis/)

---

## 📑 Table of Contents

1. [Executive Summary & Clinical Context](#1-executive-summary--clinical-context)
2. [Key Innovations & Framework Capabilities](#2-key-innovations--framework-capabilities)
3. [Deep Learning Architecture & Attention Mechanism](#3-deep-learning-architecture--attention-mechanism)
4. [Explainable AI (XAI) Interpretability Engine](#4-explainable-ai-xai-interpretability-engine)
5. [Out-of-Distribution (OOD) Input Validation](#5-out-of-distribution-ood-input-validation)
6. [Dataset Specification & Distribution](#6-dataset-specification--distribution)
7. [Empirical Benchmarks & Experimental Results](#7-empirical-benchmarks--experimental-results)
8. [Clinical Streamlit Dashboard & UI Architecture](#8-clinical-streamlit-dashboard--ui-architecture)
9. [Automated Clinical PDF Report Generator](#9-automated-clinical-pdf-report-generator)
10. [Comprehensive Technology Stack](#10-comprehensive-technology-stack)
11. [Repository Directory Structure (A to Z)](#11-repository-directory-structure-a-to-z)
12. [Developer Installation & Quick Start Guide](#12-developer-installation--quick-start-guide)
13. [CLI Model Training & Preprocessing](#13-cli-model-training--preprocessing)
14. [Streamlit Cloud Deployment Notes](#14-streamlit-cloud-deployment-notes)
15. [Research Team, Supervision & Affiliation](#15-research-team-supervision--affiliation)
16. [Medical Disclaimer & License](#16-medical-disclaimer--license)

---

## 1. Executive Summary & Clinical Context

Pulmonary diseases such as **Tuberculosis (TB)** and **Pneumonia** continue to represent major global health burdens, causing millions of preventable deaths annually—particularly in low- and middle-income countries with acute shortages of certified thoracic radiologists. While chest radiography (Chest X-Ray / CXR) is the primary frontline screening modality due to its affordability and accessibility, manual radiographic evaluation suffers from intra- and inter-observer variability, subtle overlapping opacities, and fatigue-induced diagnostic delays.

Conventional deep learning models frequently achieve high mathematical accuracy on benchmark datasets, yet they face severe barriers to real-world clinical adoption due to three fundamental issues:
1. **The "Black-Box" Dilemma:** High-accuracy predictions without transparent spatial attribution undermine physician trust.
2. **Vulnerability to Out-of-Distribution (OOD) Artifacts:** Standard neural networks blindly classify non-medical images, screenshots, or inverted artifacts with dangerously high false confidence.
3. **Isolation from Clinical Workflows:** Academic models rarely output standardized, institution-ready medical documentation containing patient metadata and visual explanations.

This project addresses these gaps by presenting a unified, clinically grounded **Hybrid CNN-CBAM Attention Architecture** coupled with multi-method XAI visualization, structural image validation, and an automated PDF medical reporting pipeline.

---

## 2. Key Innovations & Framework Capabilities

| Capability | Technical Realization | Clinical Value |
| :--- | :--- | :--- |
| **Dual-Attention CBAM Integration** | Sequential Channel & Spatial Attention modules | Suppresses background bone/soft-tissue noise; amplifies localized lesions, infiltrates, and cavitary zones. |
| **Multi-Backbone Framework** | Dynamically supports DenseNet121, ResNet50, and EfficientNet-B0 | Allows clinicians to balance sensitivity, parameter size, and inference throughput for mobile/edge or cloud tiers. |
| **Multi-Method XAI Diagnostics** | Grad-CAM, Grad-CAM++, Guided Backprop & Saliency Maps | Visualizes both coarse activation heatmaps and fine-grained anatomical/micro-textural border gradients. |
| **OOD CXR Validation Engine** | Edge Sobel SSIM matching + Spatial profile correlation | Prevents model hallucination by rejecting non-CXR images, plain graphics, or corrupted inputs before inference. |
| **Live Clinical Web Dashboard** | Streamlit with custom Glassmorphic Dark theme & step cards | Streamlined 3-step diagnostic workflow: Upload & Tune CXR ➔ Enter Patient Metadata ➔ Run Analysis. |
| **Institutional PDF Reports** | In-memory ReportLab engine with timestamps (BST / UTC+6) | Generates downloadable, print-ready clinical diagnostic certificates with patient information and XAI overlays. |
| **Leakage-Safe Benchmarking** | Unified dataset split across 13,959 multi-source CXR scans | Class-weighted Cross-Entropy loss resolves real-world dataset imbalance without artificial synthetic oversampling. |

---

## 3. Deep Learning Architecture & Attention Mechanism

The core neural engine, `HybridTBPredictor` (defined in [`src/models/hybrid_model.py`](src/models/hybrid_model.py)), couples deep convolutional feature extraction with the **Convolutional Block Attention Module (CBAM)**:

```mermaid
graph TD
    A["Raw CXR Image<br/>(3 x 224 x 224)"] --> B["OOD Validation Engine<br/>(Edge SSIM & Profile Check)"]
    B -- "Invalid Image" --> C["Reject & Display Warning"]
    B -- "Valid CXR" --> D["Pre-trained Backbone<br/>(DenseNet121 / ResNet50 / EfficientNet-B0)"]
    
    D --> E["Deep Feature Maps F<br/>[B, C, H, W]"]
    
    subgraph CBAM ["Convolutional Block Attention Module (CBAM)"]
        E --> F1["Channel Attention Module (CAM)"]
        F1 -->|"GAP + GMP + Shared MLP + σ"| F2["Channel Weights Mc(F)"]
        E --> F3["Element-wise Multiply ⊗"]
        F2 --> F3
        F3 --> G1["Channel-Refined Features F'"]
        
        G1 --> H1["Spatial Attention Module (SAM)"]
        H1 -->|"Channel Mean + Channel Max + 7x7 Conv + σ"| H2["Spatial Weights Ms(F')"]
        G1 --> H3["Element-wise Multiply ⊗"]
        H2 --> H3
        H3 --> J1["Attention-Enriched Features F''"]
    end
    
    J1 --> K["Global Average Pooling (GAP)<br/>AdaptiveAvgPool2d((1, 1))"]
    K --> L["Dense Layer (256 Neurons) + BatchNorm1d"]
    L --> M["ReLU Activation + Dropout (p=0.40)"]
    M --> N["Linear Output Layer (3 Classes)"]
    N --> O["Logits ➔ Softmax Probabilities<br/>[Normal, Tuberculosis, Pneumonia]"]
```

### 3.1 Channel Attention Module (CAM)
Channel attention addresses **"what"** semantic features are pathologically relevant. Given intermediate feature maps $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$, CAM computes spatial context descriptors using both Global Average Pooling ($\mathbf{F}_{\text{avg}}^c$) and Global Max Pooling ($\mathbf{F}_{\text{max}}^c$). These descriptors are forwarded through a shared Multi-Layer Perceptron (MLP) with a reduction ratio $r = 16$:

$$\mathbf{M}_c(\mathbf{F}) = \sigma\left(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\right) = \sigma\left(\mathbf{W}_1(\mathbf{W}_0(\mathbf{F}_{\text{avg}}^c)) + \mathbf{W}_1(\mathbf{W}_0(\mathbf{F}_{\text{max}}^c))\right)$$

Where $\mathbf{W}_0 \in \mathbb{R}^{\frac{C}{r} \times C}$ and $\mathbf{W}_1 \in \mathbb{R}^{C \times \frac{C}{r}}$ represent shared MLP layer weights, and $\sigma$ denotes the sigmoid activation function.

### 3.2 Spatial Attention Module (SAM)
Spatial attention addresses **"where"** lesions or consolidations reside within the lung fields. Channel-refined features $\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}$ are compressed along the channel axis using average and max pooling, concatenated, and convolved via a $7 \times 7$ kernel:

$$\mathbf{M}_s(\mathbf{F}') = \sigma\left(f^{7 \times 7}\left(\left[\text{AvgPool}(\mathbf{F}'); \text{MaxPool}(\mathbf{F}')\right]\right)\right) = \sigma\left(f^{7 \times 7}\left(\left[\mathbf{F}'^{s}_{\text{avg}}; \mathbf{F}'^{s}_{\text{max}}\right]\right)\right)$$

The final attention-enriched representation $\mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \otimes \mathbf{F}'$ is pooled and passed to the classification head.

---

## 4. Explainable AI (XAI) Interpretability Engine

All four XAI algorithms are implemented in PyTorch within [`app/xai_methods.py`](app/xai_methods.py) for real-time inference and report generation:

```
+-----------------------------------------------------------------------------------+
|                            EXPLAINABLE AI ENGINE                                  |
+--------------------------+--------------------------------------------------------+
| 1. Grad-CAM              | Coarse localization of class-discriminative regions     |
| 2. Grad-CAM++            | Multi-instance & weighted pixel-level attribution      |
| 3. Guided Backprop       | High-frequency structural borders & pixel textures     |
| 4. Saliency Mapping      | Raw gradient sensitivity blended with COLORMAP_HOT     |
+--------------------------+--------------------------------------------------------+
```

### 1. Grad-CAM (Gradient-Weighted Class Activation Mapping)
Computes the gradients of the predicted class score $y^c$ with respect to the last convolutional feature map $A^k$:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

*Clinical Utility:* Highlights primary focal lobes showing active consolidation or apical cavitary lesions.

### 2. Grad-CAM++
Extends Grad-CAM by computing higher-order second and third derivatives for positive weight attributions:

$$w_k^c = \sum_i \sum_j \alpha_{i,j}^{kc} \cdot \text{ReLU}\left(\frac{\partial y^c}{\partial A_{i,j}^k}\right)$$

*Clinical Utility:* Resolves co-occurring infections and multiple dispersed infiltrates across bilateral lung fields.

### 3. Guided Backpropagation
Modifies standard backpropagation by clipping negative gradients and negative feature activations during backward passes:

$$R_i^l = (R_i^{l+1} > 0) \cdot (f_i^l > 0) \cdot R_i^{l+1}$$

*Clinical Utility:* Traces crisp anatomical contours, such as pleural thickening, apical scarring, and rib borders.

### 4. Saliency Mapping
Calculates the absolute maximum gradient of the target class score relative to input pixel intensities:

$$S(x, y) = \max_c \left| \frac{\partial y^c}{\partial I(x, y, c)} \right|$$

*Clinical Utility:* Displays sensitivity hot-spots overlaid with medical thermal colormaps (`COLORMAP_HOT`).

---

## 5. Out-of-Distribution (OOD) Input Validation

To safeguard against non-radiological images (e.g., screenshots, icons, everyday objects), [`app/app.py`](app/app.py) deploys a multi-stage validation engine:

1. **Standard Deviation Thresholding ($\sigma > 15$):** Rejects uniform graphics, blank scans, and non-contrast screenshots.
2. **Sobel Edge Structural Similarity (SSIM $\ge 0.14$):** Calculates Sobel gradient maps of the input and compares structural similarity against an average anatomical CXR reference template ([`app/cxr_template.png`](app/cxr_template.png)).
3. **Anatomical Spatial Profile Correlation ($r \ge -0.15$):** Evaluates horizontal and vertical column/row mean projections to ensure the presence of standard human thoracic contours.

---

## 6. Dataset Specification & Distribution

The framework is trained and benchmarked on a curated, leak-safe hybrid dataset compiled from six international open-access radiological databases:

| Source Dataset | Primary Contribution | Samples Contributed |
| :--- | :--- | :---: |
| **OCT Chest X-Ray Repository** | Normal & Pediatric / Adult Pneumonia | 5,856 |
| **RSNA Pneumonia Challenge** | Clinical Pneumonia & Normal Scans | 3,500 |
| **NIAID TB Portal Program** | Drug-Resistant & Active Pulmonary Tuberculosis | 3,499 |
| **Shenzhen Hospital CXR Collection** | Outpatient Tuberculosis Scans (Shenzhen, China) | 662 |
| **Belarus Tuberculosis Collection** | Multidrug-Resistant Tuberculosis (MDR-TB) | 304 |
| **Montgomery County CXR Set** | Tuberculosis Control Program (Maryland, USA) | 138 |
| **Total Unified Dataset** | **Multi-Class Hybrid Repository** | **13,959** |

### Stratified Train / Validation / Test Splits:

| Diagnostic Class | Train Split (85%) | Validation Split (5%) | Test Split (10%) | Total Class Count |
| :--- | :---: | :---: | :---: | :---: |
| **Normal (Healthy)** | 4,667 | 274 | 548 | **5,489** |
| **Pneumonia** | 3,633 | 213 | 427 | **4,273** |
| **Tuberculosis** | 3,573 | 207 | 417 | **4,197** |
| **Total Samples** | **11,873** | **694** | **1,392** | **13,959** |

---

## 7. Empirical Benchmarks & Experimental Results

Evaluated on the uncorrupted hold-out **Test Set of 1,392 independent CXR scans** on an NVIDIA Tesla T4 GPU:

| Neural Backbone (Hybrid + CBAM) | Accuracy | Sensitivity (Recall) | Specificity | ROC-AUC | GPU Speed | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet121 (Hybrid)** | 97.99% | 98.06% | 98.98% | 0.9991 | 3.1 img/s | ~28.5 MB |
| **ResNet50 (Hybrid)** | 97.92% | 98.00% | 98.94% | 0.9989 | 2.2 img/s | ~98.5 MB |
| **EfficientNet-B0 (Hybrid)** ⭐ | **98.06%** | **98.18%** | **99.03%** | **0.9991** | **4.8 img/s** | **~18.5 MB** |

### Key Experimental Insights:
- **Top Performer:** **EfficientNet-B0 (Hybrid)** achieved the highest overall **Accuracy (98.06%)** and **Sensitivity (98.18%)**, making it the optimal backbone for high-sensitivity screening.
- **Throughput & Edge Viability:** EfficientNet-B0 processed **4.8 images per second**, with the smallest parameter footprint (~4.0M params), ideal for resource-constrained telemedicine units.
- **High Specificity:** All three models exceeded **98.9% Specificity**, demonstrating negligible false-positive alarms on healthy patients.

---

## 8. Clinical Streamlit Dashboard & UI Architecture

The front-end is crafted with a **Glassmorphic Dark Mode UI** ([`app/style.css`](app/style.css)) organized into three structured step cards:

```
[ Step 01: Chest X-Ray Upload ] ➔ File uploader / Webcam + Contrast & Sharpness Sliders
[ Step 02: Patient Metadata   ] ➔ Name, Age, ID, Gender, Clinical History & Symptoms
[ Step 03: Neural Model & XAI ] ➔ Backbone Selector, XAI Method, "Run Analysis" Trigger
```

### UI Highlights:
- **Auto-Scroll Focusing:** Clicking "Run Diagnostic Analysis" triggers an automated DOM scroll routine that scrolls directly to the *Diagnostic & Interpretability Report* header (with a 90px sticky navbar buffer).
- **Session Tracking Table:** An interactive, height-constrained table logging previous diagnoses during the session.
- **Model Performance Hub Tab:** Comparative double-bar charts and throughput speed benchmarks embedded natively via Matplotlib (`Agg` non-interactive backend).

---

## 9. Automated Clinical PDF Report Generator

Powered by **ReportLab** ([`app/report_generator.py`](app/report_generator.py)), the system dynamically generates institutional, print-ready diagnostic summaries in-memory:

- **Institutional Header:** DIU Department of Computer Science & Engineering branding.
- **Patient Demographics:** Patient Name, Age, ID, Gender, and Reported Symptoms.
- **Diagnostic Outcome:** Prediction badge with exact classification probability (%).
- **Dual Radiological Imagery:** Original CXR alongside the chosen XAI heatmap overlay.
- **BST / UTC+6 Timestamp:** Exact date and time formatted to Bangladesh Standard Time.
- **Sign-off Block:** Dedicated signature area for the consulting Radiologist.

---

## 10. Comprehensive Technology Stack

| Category | Technology / Library | Purpose in Project |
| :--- | :--- | :--- |
| **Programming Language** | **Python (3.10 / 3.11)** | Core programming language for models, pipelines, and web serving. |
| **Deep Learning** | **PyTorch (`torch`)** | Neural network definition, custom CBAM layers, forward/backward passes. |
| **Vision Backbones** | **Torchvision (`torchvision.models`)** | Pre-trained weights for DenseNet121, ResNet50, EfficientNet-B0. |
| **Explainable AI (XAI)** | **`grad-cam` + Custom PyTorch Hooks** | Grad-CAM, Grad-CAM++, Guided Backprop, Saliency Map generation. |
| **Computer Vision** | **OpenCV (`opencv-python-headless`)** | Edge detection, SSIM computation, colormap overlay blending. |
| **Image Manipulation** | **Pillow (`PIL`)** | Dynamic brightness, contrast, sharpness tuning, and format conversions. |
| **Data Processing** | **Pandas & NumPy** | Metadata parsing, dataset stratification, vector matrix calculations. |
| **Evaluation Metrics** | **Scikit-Learn (`scikit-learn`)** | Confusion matrix, ROC-AUC, classification report, class weights. |
| **Web Dashboard** | **Streamlit (`streamlit`)** | Reactive web UI, multi-tab navigation, session state management. |
| **Visual Styling** | **Vanilla CSS3 & HTML5** | Frosted glassmorphism, responsive grid containers, Google Poppins font. |
| **Animated Icons** | **Lordicon CDN** | Animated vector icons for loading states and step cards. |
| **PDF Generation** | **ReportLab (`reportlab`)** | Programmatic vector PDF compilation with tables, styles, and images. |
| **Server Routing** | **Starlette (`starlette<1.0.0`)** | Lightweight ASGI server components for Streamlit web hosting. |

---

## 11. Repository Directory Structure (A to Z)

```text
├── .streamlit/
│   └── config.toml               # Streamlit server and theme configurations
├── .vscode/
│   └── settings.json             # Workspace IDE settings
├── app/
│   ├── app.py                    # Main Streamlit web application & UI logic
│   ├── cxr_template.png          # Average CXR template for OOD edge SSIM validation
│   ├── diu_logo.png              # Institutional logo for PDF reports and sidebar
│   ├── report_generator.py       # ReportLab PDF report generation engine
│   ├── style.css                 # Glassmorphic dark design system stylesheet
│   └── xai_methods.py            # PyTorch XAI visualization algorithms (Grad-CAM, etc.)
├── data/
│   └── metadata.csv              # Full dataset index (13,959 CXRs with labels & splits)
├── models/
│   ├── best_tb_model_densenet121.pth       # Trained DenseNet121+CBAM weights
│   ├── best_tb_model_efficientnet_b0.pth   # Trained EfficientNet-B0+CBAM weights
│   └── best_tb_model_resnet50.pth          # Trained ResNet50+CBAM weights
├── notebooks/
│   ├── 01_data_preprocessing.ipynb         # Dataset loading, cleaning, and stratification
│   ├── 02_model_training.ipynb             # General model training experimentation
│   ├── 02_model_training_densenet121.ipynb # DenseNet121 training & evaluation
│   ├── 02_model_training_efficientnet_b0.ipynb # EfficientNet-B0 training & evaluation
│   └── 02_model_training_resnet50.ipynb    # ResNet50 training & evaluation
├── src/
│   ├── models/
│   │   └── hybrid_model.py       # PyTorch HybridTBPredictor and CBAM attention modules
│   ├── preprocess.py             # CLI metadata compilation & validation script
│   └── train.py                  # Standalone CLI model training pipeline with early stopping
├── thesis/
│   ├── 01-title-defense/         # Title phase evaluation report, slides, and approval forms
│   └── 02-phase-1/               # Phase-1 comprehensive evaluation report, slides, and diagrams
├── .gitignore                    # Git tracking rules
├── LICENSE                       # MIT Open-Source License
├── packages.txt                  # Debian system-level dependencies for cloud deployment
├── README.md                     # Project master documentation (this file)
└── requirements.txt              # Pinned Python package dependencies
```

---

## 12. Developer Installation & Quick Start Guide

### Prerequisites
- **Python:** `3.10` or `3.11` (Python 3.11 recommended)
- **Git:** Installed on your local machine
- **Hardware:** Dedicated GPU (CUDA-enabled) recommended for training; CPU is fully supported for dashboard inference.

### Step-by-Step Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Shariar-Ahamed/hybrid-tuberculosis-pneumonia-detection-xai.git
   cd hybrid-tuberculosis-pneumonia-detection-xai
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Python Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit Clinical Web Dashboard:**
   ```bash
   streamlit run app/app.py
   ```

5. **Open in Browser:** Navigate to `http://localhost:8501`.

---

## 13. CLI Model Training & Preprocessing

You can train any of the three hybrid architectures directly from the command line using [`src/train.py`](src/train.py):

### Train DenseNet121 Hybrid:
```bash
python src/train.py --backbone densenet121 --epochs 15 --batch-size 16 --lr 0.0001
```

### Train ResNet50 Hybrid:
```bash
python src/train.py --backbone resnet50 --epochs 15 --batch-size 16 --lr 0.0001
```

### Train EfficientNet-B0 Hybrid:
```bash
python src/train.py --backbone efficientnet_b0 --epochs 15 --batch-size 16 --lr 0.0001
```

### Compile / Update Metadata:
```bash
python src/preprocess.py
```

---

## 14. Streamlit Cloud Deployment Notes

When deploying to [Streamlit Community Cloud](https://streamlit.io/cloud):
- **Python Version:** Select **Python 3.11** in the Advanced Settings modal.
- **System Packages (`packages.txt`):** Contains `libgl1` to satisfy OpenCV runtime dependencies without triggering Debian Bookworm conflicts.
- **Dynamic Glib Loader:** [`app/app.py`](app/app.py) contains a runtime fallback routine to dynamically download and load `libglib-2.0` / `libgthread-2.0` via `ctypes` on Debian containers.
- **Matplotlib Backend:** Initialized with `matplotlib.use('Agg')` prior to plotting imports to avoid GUI server crashes.

---

## 15. Research Team, Supervision & Affiliation

This project was conducted as a Final Year Design Project (FYDP) / Undergraduate Thesis under the **Department of Computer Science and Engineering (CSE), Daffodil International University (DIU)**, Dhaka, Bangladesh.

### 👔 Supervision
- **Dr. Md. Ali Hossain** — Associate Professor, Department of CSE, Daffodil International University (Principal Supervisor)
- **Mr. Md. Mizanur Rahman** — Co-Supervisor, Department of CSE, Daffodil International University

### 👥 Authors & Core Developers
- **Shariar Ahamed Ripon** — Student ID: `0242310005101019`, Dept. of CSE, Daffodil International University ([GitHub Profile](https://github.com/Shariar-Ahamed))
- **Md. Moniruzzaman Rifat** — Student ID: `0242310005101020`, Dept. of CSE, Daffodil International University

---

## 16. Medical Disclaimer & License

> [!WARNING]
> **Medical Disclaimer:** This software and its underlying models are designed strictly for educational, academic, and clinical research evaluation. The outputs, prediction scores, and Explainable AI overlays generated by this system **do not constitute certified clinical diagnoses**. All findings must be reviewed and verified by a licensed healthcare professional or certified thoracic radiologist before initiating patient care.

### License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
