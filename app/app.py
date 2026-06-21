# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import os
import sys
from pathlib import Path
import time
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

# Append project root directory to path for imports
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.models.hybrid_model import HybridTBPredictor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import matplotlib.pyplot as plt
from report_generator import generate_pdf_report
from xai_methods import generate_xai_visualization

# Set page configuration with a premium icon and title
st.set_page_config(
    page_title="TB & Pneumonia XAI | Tuberculosis & Pneumonia Explainable AI Dashboard",
    page_icon="\U0001fac1",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_html(html_str):
    return " ".join(line.strip() for line in html_str.split("\n") if line.strip())


def render_performance_hub():
    st.components.v1.html(
        clean_html("""
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap" rel="stylesheet">
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background-color: transparent;
                    overflow: hidden;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    font-family: 'Poppins', sans-serif;
                }
            </style>
            <script src="https://cdn.lordicon.com/lordicon.js"></script>
            <div class="container">
                <lord-icon
                    src="https://cdn.lordicon.com/fcwmlcjj.json"
                    trigger="in"
                    delay="1500"
                    stroke="light"
                    state="in-reveal"
                    colors="primary:#121331,secondary:#a866ee,tertiary:#66ee78,quaternary:#66a1ee"
                    style="width:90px;height:90px">
                </lord-icon>
                <h2 style='color: #e2e8f0; font-weight: 700; margin-top: 5px; margin-bottom: 0; font-size: 1.6rem;'>Model Performance Hub</h2>
            </div>
        """),
        height=130
    )
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem;'>Comprehensive performance evaluation, diagnostic accuracy, and speed comparison across the three hybrid CNN model backbones.</p>", unsafe_allow_html=True)
    
    # Overview cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            clean_html("""
                <div class="metric-card" style="border: 1px solid rgba(168, 85, 247, 0.3) !important; background: rgba(168, 85, 247, 0.03) !important; height: 290px !important; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                    <div style="width: 100%; text-align: center; margin-bottom: 2px;"><span class="xai-badge" style="background-color: rgba(168, 85, 247, 0.15) !important; color: #a855f7 !important; display: inline-block;">Highly Efficient Backbone</span></div>
                    <div style="margin-top: 10px; margin-bottom: 5px; color: #e2e8f0; font-size: 1.1rem; font-weight: 600; line-height: 1.2; white-space: nowrap; text-align: center; width: 100%;">DenseNet121 (Hybrid)</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #a855f7; margin-top: 5px; margin-bottom: 5px; text-align: center; width: 100%;">97.99%</div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px; margin-bottom: 0; text-align: center; width: 100%;">Excellent feature reuse. Achieved 98.06% sensitivity and 98.98% specificity on the test dataset.</p>
                </div>
            """),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            clean_html("""
                <div class="metric-card" style="border: 1px solid rgba(56, 189, 248, 0.3) !important; background: rgba(56, 189, 248, 0.03) !important; height: 290px !important; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                    <div style="width: 100%; text-align: center; margin-bottom: 2px;"><span class="xai-badge" style="background-color: rgba(56, 189, 248, 0.15) !important; color: #38bdf8 !important; display: inline-block;">Highly Balanced Backbone</span></div>
                    <div style="margin-top: 10px; margin-bottom: 5px; color: #e2e8f0; font-size: 1.1rem; font-weight: 600; line-height: 1.2; white-space: nowrap; text-align: center; width: 100%;">ResNet50 (Hybrid)</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #38bdf8; margin-top: 5px; margin-bottom: 5px; text-align: center; width: 100%;">97.92%</div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px; margin-bottom: 0; text-align: center; width: 100%;">Stable residual mappings with high specificity (98.94%) and 98.00% sensitivity. Highly balanced results.</p>
                </div>
            """),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            clean_html("""
                <div class="metric-card" style="border: 1px solid rgba(34, 197, 94, 0.3) !important; background: rgba(34, 197, 94, 0.03) !important; height: 290px !important; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                    <div style="width: 100%; text-align: center; margin-bottom: 2px;"><span class="xai-badge" style="background-color: rgba(34, 197, 94, 0.15) !important; color: #22c55e !important; display: inline-block;">Top Accuracy & Speed</span></div>
                    <div style="margin-top: 10px; margin-bottom: 5px; color: #e2e8f0; font-size: 1.1rem; font-weight: 600; line-height: 1.2; white-space: nowrap; text-align: center; width: 100%;">EfficientNetB0 (Hybrid)</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #22c55e; margin-top: 5px; margin-bottom: 5px; text-align: center; width: 100%;">98.06%</div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px; margin-bottom: 0; text-align: center; width: 100%;">Highest overall accuracy (98.06%) and sensitivity (98.18%). Extremely lightweight with 4.8 img/s speed.</p>
                </div>
            """),
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Graphical Charts Row
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        # Plot double bar chart (Accuracy, Sensitivity, Specificity)
        fig1, ax1 = plt.subplots(figsize=(6, 4.2), facecolor='#0c0f17')
        ax1.set_facecolor('#0c0f17')
        
        models_list = ['DenseNet121', 'ResNet50', 'EfficientNetB0']
        x = np.arange(len(models_list))
        w_bar = 0.25
        
        accuracy_list = [97.99, 97.92, 98.06]
        sensitivity_list = [98.06, 98.00, 98.18]
        specificity_list = [98.98, 98.94, 99.03]
        
        rects1 = ax1.bar(x - w_bar, accuracy_list, w_bar, label='Accuracy', color='#38bdf8', edgecolor='none')
        rects2 = ax1.bar(x, sensitivity_list, w_bar, label='Sensitivity', color='#a855f7', edgecolor='none')
        rects3 = ax1.bar(x + w_bar, specificity_list, w_bar, label='Specificity', color='#4ade80', edgecolor='none')
        
        ax1.set_ylabel('Percentage (%)', color='#94a3b8', fontsize=9)
        ax1.set_title('Clinical Diagnostic Metric Comparison', color='#e2e8f0', fontsize=11, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_list, color='#cbd5e1', fontsize=8.5)
        ax1.set_ylim(90, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.1, color='#cbd5e1')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#1e293b')
        ax1.spines['bottom'].set_color('#1e293b')
        ax1.tick_params(colors='#94a3b8', labelsize=8)
        
        legend = ax1.legend(frameon=True, facecolor='#0f172a', edgecolor='#1e293b', fontsize=8)
        plt.setp(legend.get_texts(), color='#e2e8f0')
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
    with chart_col2:
        # Plot horizontal bar chart (Speed)
        fig2, ax2 = plt.subplots(figsize=(6, 4.2), facecolor='#0c0f17')
        ax2.set_facecolor('#0c0f17')
        
        speed_list = [3.1, 2.2, 4.8]  # images per second
        bars = ax2.barh(models_list, speed_list, height=0.4, color=['#a855f7', '#38bdf8', '#4ade80'], edgecolor='none')
        
        ax2.set_xlabel('Throughput (Images per Second)', color='#94a3b8', fontsize=9)
        ax2.set_title('Model Processing Speed (Throughput)', color='#e2e8f0', fontsize=11, fontweight='bold', pad=10)
        ax2.grid(axis='x', linestyle='--', alpha=0.1, color='#cbd5e1')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#1e293b')
        ax2.spines['bottom'].set_color('#1e293b')
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        
        # Add speed labels on the bars
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.15, bar.get_y() + bar.get_height()/2, f'{width:.1f} img/s', 
                    va='center', ha='left', fontsize=8, color='#cbd5e1', fontweight='bold')
                    
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
    # Table details
    st.markdown("<h4 style='color: #e2e8f0; font-family: \"Poppins\", sans-serif; font-weight: 600; margin-top: 1.5rem;'>📋 Detailed Architecture Trade-offs</h4>", unsafe_allow_html=True)
    st.markdown(
        clean_html("""
            <div class="history-table-container">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Neural Backbone</th>
                            <th>Model Parameters</th>
                            <th>Key Strengths</th>
                            <th>Key Weaknesses</th>
                            <th>Primary Clinical Use Case</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>DenseNet121 (Hybrid)</strong></td>
                            <td>~7.0M + CBAM</td>
                            <td>Excellent dense connectivity; high sensitivity (98.06%).</td>
                            <td>Higher memory footprint during backpropagation.</td>
                            <td>Standard clinical screening where recall is a high priority.</td>
                        </tr>
                        <tr>
                            <td><strong>ResNet50 (Hybrid)</strong></td>
                            <td>~23.5M + CBAM</td>
                            <td>Residual connections prevent gradient decay; high specificity (98.94%).</td>
                            <td>Larger model weight size; slower inference latency.</td>
                            <td>Confirmatory secondary audit in clinical research.</td>
                        </tr>
                        <tr>
                            <td><strong>EfficientNetB0 (Hybrid)</strong></td>
                            <td>~4.0M + CBAM</td>
                            <td>Highest overall accuracy (98.06%) & sensitivity (98.18%); lightweight & fast.</td>
                            <td>Slightly more complex scaling during initial training phases.</td>
                            <td>High-throughput remote clinics & resource-limited areas.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        """),
        unsafe_allow_html=True
    )



# Initialize session state variables for stats and history
if "session_total_scans" not in st.session_state:
    st.session_state.session_total_scans = 0
if "diagnoses_history" not in st.session_state:
    st.session_state.diagnoses_history = []
if "session_tb" not in st.session_state:
    st.session_state.session_tb = sum(1 for x in st.session_state.diagnoses_history if x.get("diagnosis") == "Tuberculosis")
if "session_pneumonia" not in st.session_state:
    st.session_state.session_pneumonia = sum(1 for x in st.session_state.diagnoses_history if x.get("diagnosis") == "Pneumonia")
if "session_normal" not in st.session_state:
    st.session_state.session_normal = sum(1 for x in st.session_state.diagnoses_history if x.get("diagnosis") == "Normal")



# Render a premium full-screen splash loading screen on the first load to hide initial setup delay
if "first_load_done" not in st.session_state:
    st.markdown(clean_html("""
        <div id="splash-screen" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: radial-gradient(circle at 50% 50%, #0c0f17 0%, #05070a 100%);
            z-index: 999999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #e2e8f0;
            font-family: 'Poppins', sans-serif;
        ">
            <style>
                @keyframes spin-splash {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                @keyframes spin-splash-reverse {
                    0% { transform: rotate(360deg); }
                    100% { transform: rotate(0deg); }
                }
                @keyframes pulse-splash-core {
                    0% { transform: scale(0.85); opacity: 0.7; }
                    100% { transform: scale(1.1); opacity: 1; }
                }
                @keyframes pulse-splash-text {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 1; }
                }
            </style>
            <div style="position: relative; margin-bottom: 1.8rem; display: flex; align-items: center; justify-content: center; width: 120px; height: 120px;">
                <!-- Outer dashed ring -->
                <div style="
                    position: absolute;
                    width: 110px;
                    height: 110px;
                    border: 2px dashed rgba(56, 189, 248, 0.3);
                    border-radius: 50%;
                    animation: spin-splash 6s linear infinite;
                "></div>
                <!-- Inner glowing ring -->
                <div style="
                    position: absolute;
                    width: 85px;
                    height: 85px;
                    border: 3px solid transparent;
                    border-top: 3px solid #38bdf8;
                    border-bottom: 3px solid #60a5fa;
                    border-radius: 50%;
                    animation: spin-splash-reverse 1.5s cubic-bezier(0.5, 0, 0.5, 1) infinite;
                    filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.5));
                "></div>
                <!-- Central pulsing core (Lungs) -->
                <div style="
                    position: absolute;
                    font-size: 2.2rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    animation: pulse-splash-core 1.2s ease-in-out infinite alternate;
                    z-index: 10;
                    filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.6));
                ">\U0001fac1</div>
            </div>
            <div style="
                font-size: 1.7rem;
                font-weight: 800;
                background: linear-gradient(135deg, #ffffff 40%, #cbd5e1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
                letter-spacing: -0.8px;
                font-family: 'Poppins', sans-serif;
            ">⚕️ TB & Pneumonia XAI Platform</div>
            <div style="
                color: #94a3b8;
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 2px;
                animation: pulse-splash-text 1.5s infinite ease-in-out;
                font-family: 'Poppins', sans-serif;
            ">Initializing Clinical Intelligence Systems...</div>
        </div>
    """), unsafe_allow_html=True)
    st.session_state.first_load_done = True

# Helper function to load local CSS files
def local_css(file_name):
    css_path = Path(__file__).resolve().parent / file_name
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Premium Custom CSS styling injection from separate stylesheet
local_css("style.css")

# Load Lordicon player script CDN for interactive animated icons
st.markdown('<script src="https://cdn.lordicon.com/lordicon.js"></script>', unsafe_allow_html=True)

from PIL import ImageEnhance

def adjust_image(image, brightness, contrast, sharpness):
    # Adjust Brightness
    enhancer = ImageEnhance.Brightness(image)
    img = enhancer.enhance(brightness)
    # Adjust Contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    # Adjust Sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness)
    return img

def reset_sliders():
    st.session_state.cxr_brightness = 1.0
    st.session_state.cxr_contrast = 1.0
    st.session_state.cxr_sharpness = 1.0

# Backward-compatible wrapper for st.image to handle version deprecations without warnings or crashes
def st_image_compatible(image, caption=None):
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)

# Load average CXR template globally for validation checks
template_path = Path(__file__).resolve().parent / "cxr_template.png"
if template_path.exists():
    CXR_TEMPLATE = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if CXR_TEMPLATE is not None:
        # Precompute template edge map
        sobelx = cv2.Sobel(CXR_TEMPLATE, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(CXR_TEMPLATE, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx**2 + sobely**2)
        CXR_TEMPLATE_EDGES = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Precompute template spatial profiles for OOD detection
        def norm_profile(p):
            return (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-6)
        CXR_TEMPLATE_COL = norm_profile(np.mean(CXR_TEMPLATE, axis=0))
        CXR_TEMPLATE_ROW = norm_profile(np.mean(CXR_TEMPLATE, axis=1))
    else:
        CXR_TEMPLATE = None
        CXR_TEMPLATE_EDGES = None
        CXR_TEMPLATE_COL = None
        CXR_TEMPLATE_ROW = None
else:
    CXR_TEMPLATE = None
    CXR_TEMPLATE_EDGES = None
    CXR_TEMPLATE_COL = None
    CXR_TEMPLATE_ROW = None

# Image validation check to prevent non-CXR images (OOD) from being analyzed
def validate_chest_xray(image):
    img_np = np.array(image)
    
    # 1. Get grayscale version for detail/contrast check
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_np
        
    # 2. Standard deviation check (filters out flat icons, plain backgrounds, screenshots)
    img_std = np.std(gray_img)
    if img_std < 15:  # Lowered to 15 to be extremely safe for low-contrast scans
        return False, "The uploaded image does not match the typical structural and anatomical patterns of a Chest X-Ray scan."
        
    # 4. Edge-based Structural Similarity (Edge SSIM) check to reject OOD non-CXR images
    if CXR_TEMPLATE_EDGES is not None:
        # Resize input to match template size (128x128)
        gray_resized = cv2.resize(gray_img, (128, 128))
        
        # Calculate Sobel edge map for the input image
        sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx**2 + sobely**2)
        input_edges = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate SSIM of the edge maps
        def compute_ssim(img1, img2):
            C1 = (0.01 * 255)**2
            C2 = (0.03 * 255)**2
            
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            
            kernel = np.ones((3, 3), np.float64) / 9.0
            mu1 = cv2.filter2D(img1, -1, kernel)
            mu2 = cv2.filter2D(img2, -1, kernel)
            
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.filter2D(img1**2, -1, kernel) - mu1_sq
            sigma2_sq = cv2.filter2D(img2**2, -1, kernel) - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
            
        edge_ssim = compute_ssim(CXR_TEMPLATE_EDGES, input_edges)
        
        # Reject image if edge structural similarity is below threshold
        if edge_ssim < 0.14:
            return False, "The uploaded image does not match the typical structural and anatomical patterns of a Chest X-Ray scan."
            
        # 5. Anatomical Profile Correlation Check to filter out OOD structures (like aeroplanes)
        if CXR_TEMPLATE_COL is not None and CXR_TEMPLATE_ROW is not None:
            def norm_profile(p):
                return (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-6)
                
            input_col = norm_profile(np.mean(gray_resized, axis=0))
            input_row = norm_profile(np.mean(gray_resized, axis=1))
            
            corr_col = np.corrcoef(CXR_TEMPLATE_COL, input_col)[0, 1]
            corr_row = np.corrcoef(CXR_TEMPLATE_ROW, input_row)[0, 1]
            
            # Loose threshold to prevent false rejections of real CXR scans, while blocking OOD shapes
            if corr_col < -0.15 or corr_row < -0.15:
                return False, "The uploaded image does not match the typical structural and anatomical patterns of a Chest X-Ray scan."
            
    return True, ""



# ----------------- MODEL LOADING (Cached) -----------------
@st.cache_resource
def load_trained_model(backbone="densenet121"):
    # Map backbones to weight files
    if backbone == "densenet121":
        # Check if the renamed file exists, otherwise fall back to old name
        if (base_dir / "models" / "best_tb_model_densenet121.pth").exists():
            weights_name = "best_tb_model_densenet121.pth"
        else:
            weights_name = "best_tb_model.pth"  # Backward compatibility
    else:
        weights_name = f"best_tb_model_{backbone}.pth"
        
    weights_path = base_dir / "models" / weights_name
    print(f"[DEBUG] load_trained_model: backbone = {backbone}")
    print(f"[DEBUG] load_trained_model: weights_path = {weights_path}")
    print(f"[DEBUG] load_trained_model: weights_path.exists() = {weights_path.exists()}")
    
    loaded = False
    if weights_path.exists():
        model = HybridTBPredictor(backbone=backbone, pretrained=False)
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            print(f"[DEBUG] load_trained_model: successfully loaded weights for {backbone}!")
            loaded = True
        except Exception as e:
            print(f"[DEBUG] load_trained_model: failed to load weights! Error: {e}")
            model = HybridTBPredictor(backbone=backbone, pretrained=True)
    else:
        print(f"[DEBUG] load_trained_model: weights file not found for {backbone}. Fallback to pre-trained ImageNet weights.")
        model = HybridTBPredictor(backbone=backbone, pretrained=True)
        
    model.eval()
    return model, loaded
# Transforms for inference (exactly same as validation transforms)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- SIDEBAR CONTENT -----------------
with st.sidebar:
    logo_path = Path(__file__).resolve().parent / "diu_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=170)
    else:
        st.image("https://i.postimg.cc/HnfyLQYZ/DIU-Logo.png", width=170)
    st.markdown("<hr style='margin: 8px 0; opacity: 0.15;'>", unsafe_allow_html=True)
        
    st.markdown("### FYDP Title Phase Evaluation")
    st.markdown("<hr style='margin: 8px 0; opacity: 0.15;'>", unsafe_allow_html=True)
    
    st.markdown("#### **Project Title**")
    st.markdown(
        "<div class='sidebar-card' style='font-size: 0.9rem; font-weight: 500;'>"
        "A Hybrid Explainable AI Framework for Automatic Tuberculosis and Pneumonia Detection from Chest X-Ray Images Using Convolutional Neural Networks"
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown("#### **Team Members**")
    st.markdown(
        "• **Shariar Ahamed Ripon**<br><small style='color:#64748b;'>ID: 0242310005101019</small><br>"
        "• **Md. Moniruzzaman Rifat**<br><small style='color:#64748b;'>ID: 0242310005101020</small>",
        unsafe_allow_html=True
    )
    
    st.markdown("#### **Supervision**")
    st.markdown(
        "• **Dr. Md. Ali Hossain** (Associate Professor)<br>"
        "• **Mr. Md. Mizanur Rahman** (Co-Supervisor)",
        unsafe_allow_html=True
    )


# ----------------- MAIN VIEW -----------------
# Header Banner
st.markdown("""
<div class="header-banner">
    <div class="header-flex">
        <div class="header-icon-container">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="header-svg">
                <path d="M8.5 1.5a.5.5 0 1 0-1 0v5.243L7 7.1V4.72C7 3.77 6.23 3 5.28 3c-.524 0-1.023.27-1.443.592-.431.332-.847.773-1.216 1.229-.736.908-1.347 1.946-1.58 2.48-.176.405-.393 1.16-.556 2.011-.165.857-.283 1.857-.241 2.759.04.867.233 1.79.838 2.33.67.6 1.622.556 2.741-.004l1.795-.897A2.5 2.5 0 0 0 7 11.264V10.5a.5.5 0 0 0-1 0v.764a1.5 1.5 0 0 1-.83 1.342l-1.794.897c-.978.489-1.415.343-1.628.152-.28-.25-.467-.801-.505-1.63-.037-.795.068-1.71.224-2.525.157-.82.357-1.491.491-1.8.19-.438.75-1.4 1.44-2.25.342-.422.703-.799 1.049-1.065.358-.276.639-.385.833-.385a.72.72 0 0 1 .72.72v3.094l-1.79 1.28a.5.5 0 0 0 .58.813L8 7.614l3.21 2.293a.5.5 0 1 0 .58-.814L10 7.814V4.72a.72.72 0 0 1 .72-.72c.194 0 .475.11.833.385.346.266.706.643 1.05 1.066.688.85 1.248 1.811 1.439 2.249.134.309.334.98.491 1.8.156.814.26 1.73.224 2.525-.038.829-.224 1.38-.505 1.63-.213.19-.65.337-1.628-.152l-1.795-.897A1.5 1.5 0 0 1 10 11.264V10.5a.5.5 0 0 0-1 0v.764a2.5 2.5 0 0 0 1.382 2.236l1.795.897c1.12.56 2.07.603 2.741.004.605-.54.798-1.463.838-2.33.042-.902-.076-1.902-.24-2.759-.164-.852-.38-1.606-.558-2.012-.232-.533-.843-1.571-1.579-2.479-.37-.456-.785-.897-1.216-1.229C11.743 3.27 11.244 3 10.72 3 9.77 3 9 3.77 9 4.72V7.1l-.5-.357z"/>
            </svg>
        </div>
        <div class="header-text-container">
            <span class="header-badge">AI Clinical Assistant</span>
            <div class="header-title">Hybrid Explainable AI Framework</div>
            <div class="header-subtitle">Automatic Tuberculosis & Pneumonia Detection with Explainable AI (XAI) Visualizations from Chest X-Ray Images</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Metrics Section
# Stats Metrics Section (Responsive CSS Grid containers for Model Metrics & Live Session Stats)
total_scans = st.session_state.session_total_scans
tb_count = st.session_state.session_tb
pneumonia_count = st.session_state.session_pneumonia
normal_count = st.session_state.session_normal

current_xai = st.session_state.get("xai_method_select", "Grad-CAM")
xai_display = "Guided BP" if current_xai == "Guided Backpropagation" else current_xai

st.markdown(
    clean_html(f"""
        <div class="metrics-container" style="margin-bottom: 0.85rem;">
            <div class="metric-card">
                <div class="metric-value">98.06%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{xai_display}</div>
                <div class="metric-label">XAI Engine</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">0.45s</div>
                <div class="metric-label">Inference Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">13,959 Scans</div>
                <div class="metric-label">Dataset Size</div>
            </div>
        </div>
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-value">{total_scans}</div>
                <div class="metric-label">Total Scans Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-value-tb">{tb_count}</div>
                <div class="metric-label">Tuberculosis Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-value-pneumonia">{pneumonia_count}</div>
                <div class="metric-label">Pneumonia Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-value-normal">{normal_count}</div>
                <div class="metric-label">Normal Detected</div>
            </div>
        </div>
    """), 
    unsafe_allow_html=True
)







# File Uploader and Diagnostic Display

# Create tabs for Diagnostics and Performance Comparison
tab_diag, tab_perf = st.tabs(["🔬 Clinical Diagnostics & XAI", "📊 Model Performance Hub"])

with tab_diag:
    upload_col, display_col = st.columns([1, 2])

    with upload_col:
        st.components.v1.html(
            """
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background-color: transparent;
                    overflow: hidden;
                }
                .header-container {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-family: 'Poppins', sans-serif;
                }
            </style>
            <script src="https://cdn.lordicon.com/lordicon.js"></script>
            <div class="header-container">
                <lord-icon
                    src="https://cdn.lordicon.com/rqedipls.json"
                    trigger="in"
                    delay="1500"
                    state="in-reveal"
                    colors="primary:#17171c,secondary:#66a1ee,tertiary:#ffffff"
                    style="width:36px;height:36px;">
                </lord-icon>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #e2e8f0; line-height: 36px;">Upload CXR Scan</h3>
            </div>
            """,
            height=40
        )
    
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0
        
        if "image_source" not in st.session_state:
            st.session_state.image_source = "📄 File Upload"
        
        if "mode" in st.query_params:
            mode = st.query_params["mode"]
            if mode == "camera":
                st.session_state.image_source = "📷 Use Camera"
                st.query_params.clear()
                st.rerun()
            elif mode == "upload":
                st.session_state.image_source = "📄 File Upload"
                st.query_params.clear()
                st.rerun()
        
        if st.session_state.image_source == "📄 File Upload":
            uploaded_file = st.file_uploader(
                "Upload a Chest X-ray image (PNG or JPG format) to detect Tuberculosis and Pneumonia signs.", 
                type=["png", "jpg", "jpeg"],
                key=f"file_uploader_{st.session_state.uploader_key}"
            )
            if st.button("📷 Use Camera", key="trigger_camera_mode", help="Switch to webcam input mode to capture a scan directly."):
                st.session_state.image_source = "📷 Use Camera"
                st.rerun()
        else:
            uploaded_file = st.camera_input(
                "Capture a grayscale Chest X-Ray scan image using your webcam or camera.",
                key=f"camera_input_{st.session_state.uploader_key}"
            )
            if st.button("📄 Switch to File Upload", key="trigger_upload_mode", help="Switch back to uploading a scan from your local files."):
                st.session_state.image_source = "📄 File Upload"
                st.rerun()

        # Perform validation immediately to render error card before Model Configuration if needed
        is_valid = True
        err_msg = ""
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            is_valid, err_msg = validate_chest_xray(image)
            if not is_valid:
                st.components.v1.html(
                    f"""
                    <style>
                        body {{
                            margin: 0;
                            padding: 0;
                            background-color: transparent;
                            overflow: hidden;
                        }}
                        .error-card {{
                            display: flex;
                            align-items: center;
                            gap: 12px;
                            background-color: rgba(244, 69, 69, 0.08);
                            border: 1px solid rgba(244, 69, 69, 0.2);
                            border-radius: 8px;
                            padding: 8px 14px;
                            color: #f87171;
                            font-family: 'Poppins', sans-serif;
                        }}
                    </style>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                    <div class="error-card">
                        <i class="fa-solid fa-xmark fa-beat-fade fa-xl" style="color: rgb(244, 69, 69); margin-left: 4px; margin-right: 4px;"></i>
                        <span style="font-size: 0.93rem; font-weight: 500; line-height: 22px;">Image Validation Failed: {err_msg}</span>
                    </div>
                    """,
                    height=115
                )
                st.warning("Please upload a genuine grayscale Chest X-Ray scan to proceed.")

        # Model Configuration on Main Page
        st.markdown("<hr style='margin: 15px 0; opacity: 0.1;'>", unsafe_allow_html=True)
        st.markdown("#### **🧠 Model Configuration**")
        backbone_choice = st.selectbox(
            "Select Neural Backbone",
            ["DenseNet121 (Hybrid)", "ResNet50 (Hybrid)", "EfficientNetB0 (Hybrid)"],
            index=0,
            key="model_backbone_select"
        )
    
        backbone_map = {
            "DenseNet121 (Hybrid)": "densenet121",
            "ResNet50 (Hybrid)": "resnet50",
            "EfficientNetB0 (Hybrid)": "efficientnet_b0"
        }
        selected_backbone = backbone_map[backbone_choice]
    
        # Placeholder for weights loading status
        model_status_placeholder = st.empty()
    
        if uploaded_file is not None:
            if not is_valid:
                analyze_button = False
            else:
                # --- Image Enhancement Settings ---
                with st.expander("🎛️ CXR Image Enhancement Settings", expanded=False):
                    if "cxr_brightness" not in st.session_state:
                        st.session_state.cxr_brightness = 1.0
                    if "cxr_contrast" not in st.session_state:
                        st.session_state.cxr_contrast = 1.0
                    if "cxr_sharpness" not in st.session_state:
                        st.session_state.cxr_sharpness = 1.0
                
                    brightness = st.slider("Brightness", 0.5, 2.0, key="cxr_brightness", step=0.05)
                    contrast = st.slider("Contrast", 0.5, 2.0, key="cxr_contrast", step=0.05)
                    sharpness = st.slider("Sharpness", 0.0, 3.0, key="cxr_sharpness", step=0.05)
                
                    st.button("🔄 Reset to Default Settings", on_click=reset_sliders, use_container_width=True, type="secondary")
                
                display_image = adjust_image(image, brightness, contrast, sharpness)
            
                # Display small preview of uploaded image
                st_image_compatible(display_image, caption="Uploaded Scan Preview (Adjusted)")
            
                st.components.v1.html(
                    """
                    <style>
                        body {
                            margin: 0;
                            padding: 0;
                            background-color: transparent;
                            overflow: hidden;
                        }
                        .success-card {
                            display: flex;
                            align-items: center;
                            gap: 12px;
                            background-color: rgba(22, 199, 46, 0.08);
                            border: 1px solid rgba(22, 199, 46, 0.2);
                            border-radius: 8px;
                            padding: 8px 14px;
                            color: #4ade80;
                            font-family: 'Poppins', sans-serif;
                        }
                    </style>
                    <script src="https://cdn.lordicon.com/lordicon.js"></script>
                    <div class="success-card">
                        <lord-icon
                            src="https://cdn.lordicon.com/pxixoqxa.json"
                            trigger="in"
                            delay="1500"
                            state="in-reveal"
                            colors="primary:#121331,secondary:#16c72e"
                            style="width:26px;height:26px;flex-shrink:0;">
                        </lord-icon>
                        <span style="font-size: 0.93rem; font-weight: 500; line-height: 22px; margin-left: 4px;">Valid Chest X-Ray scan detected!</span>
                    </div>
                    """,
                    height=65
                )
            
                # --- Patient Information Inputs ---
                st.markdown("<hr style='margin: 15px 0; opacity: 0.1;'>", unsafe_allow_html=True)
                st.markdown("#### 📋 Patient Metadata")
            
                pat_col1, pat_col2 = st.columns(2)
                with pat_col1:
                    patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
                    patient_age = st.text_input("Age", placeholder="e.g. 45")
                with pat_col2:
                    patient_id = st.text_input("Patient ID / Ref", placeholder="e.g. ID-9843")
                    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                
                patient_symptoms = st.text_area(
                    "Symptoms & Clinical History", 
                    placeholder="e.g. Cough for 3 weeks, fever, night sweats...",
                    height=70
                )
                # --- XAI Method Selection ---
                st.markdown("<hr style='margin: 15px 0; opacity: 0.1;'>", unsafe_allow_html=True)
                st.markdown("#### 🔍 Explainable AI (XAI) Settings")
                xai_method = st.selectbox(
                    "Select Interpretability Method",
                    ["Grad-CAM", "Grad-CAM++", "Guided Backpropagation", "Saliency Map"],
                    help="Choose the algorithm used to visualize which features or regions the neural network focused on.",
                    key="xai_method_select"
                )
                st.markdown("<hr style='margin: 15px 0; opacity: 0.1;'>", unsafe_allow_html=True)
            
                # Trigger prediction and clear buttons in columns
                btn_col1, btn_col2 = st.columns([1.35, 0.65])
                with btn_col1:
                    analyze_button = st.button("Run Diagnostic Analysis", use_container_width=True, type="primary")
                with btn_col2:
                    clear_button = st.button("Clear Report", use_container_width=True, type="secondary")
                
                if clear_button:
                    st.session_state["uploader_key"] += 1
                    st.rerun()
        else:
            st.info("Waiting for Chest X-Ray upload...")
            analyze_button = False

    # ----------------- DEFERRED MODEL LOADING -----------------
    # Load model after layout renders to prevent blocking blank screen on initial page refresh
    with st.spinner(f"🧠 Loading {backbone_choice}..."):
        model, model_loaded = load_trained_model(selected_backbone)

    if model_loaded:
        with model_status_placeholder:
            st.components.v1.html(
                f"""
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        background-color: transparent;
                        overflow: hidden;
                    }}
                    .success-card {{
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        background-color: rgba(22, 199, 46, 0.08);
                        border: 1px solid rgba(22, 199, 46, 0.2);
                        border-radius: 8px;
                        padding: 8px 14px;
                        color: #4ade80;
                        font-family: 'Poppins', sans-serif;
                    }}
                </style>
                <script src="https://cdn.lordicon.com/lordicon.js"></script>
                <div class="success-card">
                    <lord-icon
                        src="https://cdn.lordicon.com/pxixoqxa.json"
                        trigger="in"
                        delay="1500"
                        state="in-reveal"
                        colors="primary:#121331,secondary:#16c72e"
                        style="width:26px;height:26px;flex-shrink:0;">
                    </lord-icon>
                    <span style="font-size: 0.93rem; font-weight: 500; line-height: 22px; margin-left: 4px;">Trained {backbone_choice} weights loaded successfully!</span>
                </div>
                """,
                height=65
            )
    else:
        model_status_placeholder.warning(f"⚠️ {backbone_choice} weights not found. Running with pre-trained ImageNet backbone.")

    with display_col:
        st.subheader("🔬 Diagnostic & Interpretability Report")
    
        if uploaded_file is not None and analyze_button:
            loader_placeholder = st.empty()
        
            # UI animation steps with custom premium spinner updates
            for i in range(100):
                time.sleep(0.015)  # Creates a smooth 1.5-second visual workflow
                if i < 30:
                    status = "Initializing Hybrid CNN Feature Fusion..."
                    icon = "🔄"
                elif i < 70:
                    status = "Feeding features to Spatial Attention Blocks..."
                    icon = "🧠"
                else:
                    status = f"Generating {xai_method} Interpretability Maps..."
                    icon = "⚡"
                
                loader_placeholder.markdown(f"""
                <div class="premium-loader-box">
                    <div class="scanner-container">
                        <div class="scanner-outer"></div>
                        <div class="scanner-inner"></div>
                        <div class="scanner-core">{icon}</div>
                    </div>
                    <div class="loader-title">
                        <span class="loader-text">Processing Diagnostic Analysis</span>
                    </div>
                    <div class="loader-desc">{status} ({i}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show a transition state during the actual deep learning computation to prevent blank UI state
            loader_placeholder.markdown("""
            <div class="premium-loader-box">
                <div class="scanner-container">
                    <div class="scanner-outer"></div>
                    <div class="scanner-inner"></div>
                    <div class="scanner-core">🧠</div>
                </div>
                <div class="loader-title">
                    <span class="loader-text">Executing Model Inference</span>
                </div>
                <div class="loader-desc">Running deep learning classification and generating XAI explanations...</div>
            </div>
            """, unsafe_allow_html=True)
        
            # ---- REAL MODEL INFERENCE ----
            # 1. Preprocess input image
            input_tensor = inference_transform(image).unsqueeze(0)
        
            # 2. Forward pass through model
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
                class_idx = int(torch.argmax(outputs, dim=1).item())
                class_names = ["Normal", "Tuberculosis", "Pneumonia"]
                prediction_name = class_names[class_idx]
                confidence = probabilities[class_idx] * 100
            
            # Update session stats and log diagnosis details in history
            st.session_state.session_total_scans += 1
            if prediction_name == "Tuberculosis":
                st.session_state.session_tb += 1
            elif prediction_name == "Pneumonia":
                st.session_state.session_pneumonia += 1
            else:
                st.session_state.session_normal += 1


            st.session_state.diagnoses_history.append({
                "date_time": time.strftime("%d-%b-%Y, %I:%M %p"),
                "timestamp": time.time(),
                "patient_id": patient_id or "N/A",
                "patient_name": patient_name or "Anonymous",
                "patient_gender": patient_gender or "N/A",
                "diagnosis": prediction_name,
                "confidence": f"{confidence:.2f}%",
                "model": backbone_choice.split()[0],
                "xai_method": xai_method
            })


            
            # 3. Generate XAI interpretability map
            try:
                overlay = generate_xai_visualization(
                    method_name=xai_method,
                    model=model,
                    input_tensor=input_tensor,
                    original_image=display_image
                )
            except Exception as e:
                overlay = None
                st.error(f"Could not generate {xai_method} explanation: {e}")
            
            # Clear the loader only after the model processing is completely finished
            loader_placeholder.empty()
        
            # Show results side by side
            img_col1, img_col2 = st.columns(2)
        
            # Adjusted Image
            with img_col1:
                st_image_compatible(display_image.resize((224, 224)), caption="Adjusted CXR View")
            
            # Selected XAI visualization image display
            with img_col2:
                if overlay is not None:
                    st_image_compatible(overlay, caption=f"{xai_method} Interpretability Map")
                else:
                    st.warning("Visualization display not available")
            
            # Diagnostic Details Card
            p_normal = probabilities[0] * 100
            p_tb = probabilities[1] * 100
            p_pneumonia = probabilities[2] * 100

            if prediction_name == "Tuberculosis":
                badge_html = "<span class='badge-positive'>Tuberculosis Detected</span>"
                insights_html = (
                    "• The attention block shows high activation (highlighted in red) over <b>lesion and infiltrate areas</b>.<br>"
                    "• High correlation detected with consolidations in upper lung fields."
                )
                card_class = "result-card-positive"
            elif prediction_name == "Pneumonia":
                badge_html = "<span class='badge-warning'>Pneumonia Detected</span>"
                insights_html = (
                    "• The attention block highlights <b>diffuse opacity or consolidations</b> typical of pneumonia.<br>"
                    "• High concentration of localized gradients in lower or mid lung fields."
                )
                card_class = "result-card-warning"
            else:
                badge_html = "<span class='badge-negative'>Normal (No Pathology Detected)</span>"
                insights_html = (
                    "• Clear lung fields with standard dark background.<br>"
                    "• Attention maps show no suspicious activation clusters or consolidations."
                )
                card_class = "result-card-negative"
            
            st.markdown(
                clean_html(f"""
                    <div class="report-card {card_class}">
                        <h3 style="margin-top:0; color:#e2e8f0; font-family:'Poppins',sans-serif; font-size:1.4rem;">🔬 Diagnostic & Interpretability Report</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-top: 1.2rem;">
                            <div>
                                <h5 style="margin-top:0; margin-bottom:0.5rem; color:#94a3b8; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">Prediction Status</h5>
                                {badge_html}
                                <p style="margin-top:1.2rem; font-size:1.1rem; color:#e2e8f0; margin-bottom:0;">Confidence Score: <strong>{confidence:.2f}%</strong></p>
                            </div>
                            <div>
                                <h5 style="margin-top:0; margin-bottom:0.5rem; color:#94a3b8; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">{xai_method} Insights</h5>
                                <div style="font-size:0.95rem; line-height:1.6; color:#cbd5e1;">{insights_html}</div>
                            </div>
                        </div>
                        <div style="margin-top: 1.5rem; background: rgba(0,0,0,0.15); padding: 1.2rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.03);">
                            <h5 style="margin-top:0; margin-bottom:1rem; color:#94a3b8; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;">Probability Distribution</h5>
                            <!-- Normal -->
                            <div style="margin-bottom: 0.8rem;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem;">
                                    <span style="color: #cbd5e1; font-weight: 500;">Normal (Healthy)</span>
                                    <span style="color: #4ade80; font-weight: 700;">{p_normal:.2f}%</span>
                                </div>
                                <div style="background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                                    <div style="background: linear-gradient(90deg, #22c55e, #4ade80); width: {p_normal}%; height: 100%; border-radius: 4px;"></div>
                                </div>
                            </div>
                            <!-- TB -->
                            <div style="margin-bottom: 0.8rem;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem;">
                                    <span style="color: #cbd5e1; font-weight: 500;">Tuberculosis</span>
                                    <span style="color: #f87171; font-weight: 700;">{p_tb:.2f}%</span>
                                </div>
                                <div style="background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                                    <div style="background: linear-gradient(90deg, #ef4444, #f87171); width: {p_tb}%; height: 100%; border-radius: 4px;"></div>
                                </div>
                            </div>
                            <!-- Pneumonia -->
                            <div style="margin-bottom: 0.2rem;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem;">
                                    <span style="color: #cbd5e1; font-weight: 500;">Pneumonia</span>
                                    <span style="color: #fb923c; font-weight: 700;">{p_pneumonia:.2f}%</span>
                                </div>
                                <div style="background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                                    <div style="background: linear-gradient(90deg, #f97316, #fb923c); width: {p_pneumonia}%; height: 100%; border-radius: 4px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                """), 
                unsafe_allow_html=True
            )
        
            # Generate XAI interpretability map for PDF report using original raw image
            try:
                pdf_overlay = generate_xai_visualization(
                    method_name=xai_method,
                    model=model,
                    input_tensor=input_tensor,
                    original_image=image
                )
            except Exception as e:
                pdf_overlay = None

            # Generate the clinical PDF report bytes
            pdf_data = generate_pdf_report(
                patient_name=patient_name,
                patient_id=patient_id,
                patient_age=patient_age,
                patient_gender=patient_gender,
                symptoms=patient_symptoms,
                prediction=prediction_name,
                confidence=confidence,
                insights=insights_html,
                original_image=image,
                xai_overlay=pdf_overlay,
                xai_method_name=xai_method
            )
        
            # Display PDF download button styled as a premium button
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Generate & Download Clinical PDF Report",
                data=pdf_data,
                file_name=f"TB_Pneumonia_Diagnostic_Report_{patient_id or 'Patient'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        else:
            st.markdown(
                clean_html("""
                    <div class="welcome-container">
                        <div class="welcome-header">
                            <span class="welcome-status-dot"></span>
                            <span class="welcome-status-text">System Active & Ready for Diagnostic Execution</span>
                        </div>
                        <div class="welcome-main-card">
                            <div class="welcome-icon">\U0001fac1</div>
                            <h2>No Active Diagnostic Report</h2>
                            <p>Please upload a grayscale Chest X-Ray scan on the left column, fill in the patient metadata, and click <strong>'Run Diagnostic Analysis'</strong> to trigger the AI classifier model and interpretability engine.</p>
                        </div>
                        <h4 class="capabilities-title">🩺 Clinical Intelligence Capabilities</h4>
                        <div class="capabilities-grid">
                            <div class="capability-card">
                                <div class="cap-icon-box" style="background: rgba(168, 85, 247, 0.1); color: #a855f7;">🔍</div>
                                <h5>Automatic CXR Validation</h5>
                                <p>Our OOD validation engine uses Edge SSIM and profile correlation to filter out non-lung images.</p>
                            </div>
                            <div class="capability-card">
                                <div class="cap-icon-box" style="background: rgba(56, 189, 248, 0.1); color: #38bdf8;">🧠</div>
                                <h5>Multi-Class Diagnostics</h5>
                                <p>DenseNet121 based feature extraction classifies Normal, Tuberculosis, and Pneumonia scans.</p>
                            </div>
                            <div class="capability-card">
                                <div class="cap-icon-box" style="background: rgba(234, 179, 8, 0.1); color: #eab308;">⚡</div>
                                <h5>Explainable AI Engine</h5>
                                <p>Visualizes neural pathways and decision zones via Grad-CAM, Saliency, and Guided Backprop.</p>
                            </div>
                            <div class="capability-card">
                                <div class="cap-icon-box" style="background: rgba(34, 197, 94, 0.1); color: #22c55e;">📥</div>
                                <h5>Clinical PDF Exports</h5>
                                <p>Generates signed, institutional clinical reports containing patient data and XAI overlays.</p>
                            </div>
                        </div>
                    </div>
                """), 
                unsafe_allow_html=True
            )


    # Render the recent diagnoses session history if we have records
    if st.session_state.diagnoses_history:
        st.markdown("<hr style='margin: 40px 0 20px 0; opacity: 0.15;'>", unsafe_allow_html=True)
        st.markdown("### 🕒 Recent Diagnoses Session History")
    
        # Build a beautiful HTML table for the history
        table_rows = ""
    
        # Get user's timezone from browser context, default to Dhaka timezone
        user_tz_str = st.context.timezone or "Asia/Dhaka"
        try:
            user_tz = ZoneInfo(user_tz_str)
        except Exception:
            user_tz = ZoneInfo("Asia/Dhaka")

        for idx, item in enumerate(reversed(st.session_state.diagnoses_history)): # Show newest first
            # Select badge class based on diagnosis
            if item["diagnosis"] == "Normal":
                badge_class = "history-badge-normal"
            elif item["diagnosis"] == "Tuberculosis":
                badge_class = "history-badge-tb"
            else:
                badge_class = "history-badge-pneumonia"
            
            # Safe extractions for older records
            timestamp = item.get("timestamp", None)
            if timestamp:
                try:
                    dt_obj = datetime.fromtimestamp(timestamp, tz=user_tz)
                    dt = dt_obj.strftime("%d-%b-%Y, %I:%M %p")
                except Exception:
                    dt = item.get("date_time", item.get("time", "N/A"))
            else:
                dt = item.get("date_time", item.get("time", "N/A"))
            
            gender = item.get("patient_gender", "N/A")
            model_name = item.get("model", "N/A")
        
            table_rows += clean_html(f"""
                <tr>
                    <td>{dt}</td>
                    <td><code>{item["patient_id"]}</code></td>
                    <td>{item["patient_name"]}</td>
                    <td>{gender}</td>
                    <td><span class="history-badge {badge_class}">{item["diagnosis"]}</span></td>
                    <td><strong>{item["confidence"]}</strong></td>
                    <td><span class="xai-badge" style="background-color: rgba(168, 85, 247, 0.08) !important; color: #a855f7 !important; border-color: rgba(168, 85, 247, 0.2) !important;">{model_name}</span></td>
                    <td><span class="xai-badge">{item["xai_method"]}</span></td>
                </tr>
            """) + "\n"
        
        st.markdown(
            clean_html(f"""
                <div class="history-table-container">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Date & Time</th>
                                <th>Patient ID</th>
                                <th>Patient Name</th>
                                <th>Gender</th>
                                <th>Diagnosis Result</th>
                                <th>Confidence Score</th>
                                <th>Model</th>
                                <th>XAI Method</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            """), 
            unsafe_allow_html=True
        )



with tab_perf:
    render_performance_hub()

# Render the premium footer at the bottom of the page
st.markdown("""

<div class="dashboard-footer">
    <div class="footer-left">
        <span class="footer-left-title">\U0001fac1 TB-XAI Platform</span>
        <span>|</span>
        <span>© 2026 Daffodil International University</span>
    </div>
    <div class="footer-center">
        <b>Medical Disclaimer:</b> This dashboard is a prototype for educational and research evaluation. 
        All AI diagnostic outputs and Grad-CAM visualizations must be validated by certified radiologists.
    </div>
    <div class="footer-right">
        <a href="https://github.com/Shariar-Ahamed/hybrid-tuberculosis-pneumonia-detection-xai" target="_blank" class="footer-link">
            <svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
            GitHub Codebase
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hide the splash screen once loading and layout rendering are complete
st.markdown(
    clean_html("""
        <style>
            #splash-screen {
                display: none !important;
            }
        </style>
    """), 
    unsafe_allow_html=True
)
# Trigger auto-reload for header banner top spacing fixes

