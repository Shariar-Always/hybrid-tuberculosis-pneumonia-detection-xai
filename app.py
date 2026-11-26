# -*- coding: utf-8 -*-
"""
Hugging Face Native Clinical Web Application (Gradio)
for Automatic Tuberculosis & Pneumonia Detection with Explainable AI (XAI).
"""

import os
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.hybrid_model import HybridTBPredictor
from app.xai_methods import generate_xai_visualization
from app.report_generator import generate_pdf_report

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache models in memory to avoid reloading on each inference
LOADED_MODELS = {}

def get_cached_model(backbone="efficientnet_b0", use_cbam=True):
    key = (backbone, use_cbam)
    if key not in LOADED_MODELS:
        model = HybridTBPredictor(backbone=backbone, pretrained=False, use_cbam=use_cbam)
        weights_path = ROOT_DIR / "models" / f"best_tb_model_{backbone}.pth"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[INFO] Loaded weights for {backbone} (CBAM={use_cbam})")
        else:
            print(f"[WARNING] Checkpoint not found at {weights_path}, using initialized weights.")
        model.to(DEVICE)
        model.eval()
        LOADED_MODELS[key] = model
    return LOADED_MODELS[key]


def predict_cxr(image, backbone, use_cbam, patient_name, patient_id, age, gender, symptoms):
    """Executes end-to-end diagnostic inference, XAI overlays, and PDF compilation."""
    if image is None:
        raise gr.Error("Please upload a Chest X-Ray image first.")
        
    pil_image = image.convert("RGB")
    input_tensor = INFERENCE_TRANSFORMS(pil_image).unsqueeze(0).to(DEVICE)
    
    # 1. Forward Pass
    model = get_cached_model(backbone=backbone, use_cbam=use_cbam)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    pred_idx = int(probs.argmax())
    predicted_class = CLASS_NAMES[pred_idx]
    confidence_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    top_confidence_pct = float(probs[pred_idx]) * 100.0
    
    # 2. Generate 4 Explainable AI (XAI) Visualizations
    grad_cam_img = generate_xai_visualization("Grad-CAM", model, input_tensor, pil_image)
    grad_cam_pp_img = generate_xai_visualization("Grad-CAM++", model, input_tensor, pil_image)
    guided_bp_img = generate_xai_visualization("Guided Backpropagation", model, input_tensor, pil_image)
    saliency_img = generate_xai_visualization("Saliency Map", model, input_tensor, pil_image)
    
    # 3. Generate Clinical Diagnostic PDF Report
    insights = (
        f"- Automated Screening Finding: {predicted_class} (Confidence: {top_confidence_pct:.2f}%).\n"
        f"- Architecture: {backbone.upper()} with {'CBAM Channel and Spatial Attention' if use_cbam else 'Baseline Extraction'}.\n"
        f"- Explainable AI (XAI) Attributions: Pathological opacities highlighted in pulmonary fields.\n"
        f"- Clinical Recommendation: Correlate with clinical sputum culture and GeneXpert MTB/RIF assay."
    )
    
    pdf_bytes = generate_pdf_report(
        patient_name=patient_name or "Anonymous Patient",
        patient_id=patient_id or "CXR-REF-001",
        patient_age=str(age) if age else "Unspecified",
        patient_gender=gender or "Unspecified",
        symptoms=symptoms or "None reported",
        prediction=predicted_class,
        confidence=top_confidence_pct,
        insights=insights,
        original_image=pil_image,
        xai_overlay=grad_cam_pp_img,
        xai_method_name="Grad-CAM++",
    )
    
    # Save PDF to temporary file for user download
    temp_dir = tempfile.gettempdir()
    safe_id = (patient_id or "patient").replace(" ", "_").replace("/", "_")
    pdf_file_path = os.path.join(temp_dir, f"Diagnostic_Report_{safe_id}.pdf")
    with open(pdf_file_path, "wb") as f:
        f.write(pdf_bytes)
        
    return (
        confidence_dict,
        grad_cam_img,
        grad_cam_pp_img,
        guided_bp_img,
        saliency_img,
        pdf_file_path
    )


# Custom UI CSS & Theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, title="TB & Pneumonia Detection | Hugging Face XAI") as demo:
    gr.Markdown(
        """
        # 🫁 Hybrid Explainable AI Platform for TB & Pneumonia Screening
        ### High-Precision Multi-Class Classification (Normal, Pneumonia, Tuberculosis) with CBAM Attention & XAI
        *Trained and evaluated across 22,356 multi-center digital chest radiographs.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            image_input = gr.Image(type="pil", label="Upload Digital Chest Radiograph (CXR)")
            
            with gr.Accordion("⚙️ Model Architecture & Attention Settings", open=True):
                backbone_choice = gr.Dropdown(
                    choices=["efficientnet_b0", "densenet121", "resnet50"],
                    value="efficientnet_b0",
                    label="Backbone Architecture"
                )
                cbam_toggle = gr.Checkbox(
                    value=True,
                    label="Enable CBAM Attention Module (Channel + Spatial)"
                )
                
            with gr.Accordion("📋 Patient & Clinical Demographics", open=True):
                with gr.Row():
                    patient_name = gr.Textbox(label="Patient Name", value="Anonymous Patient")
                    patient_id = gr.Textbox(label="Patient / Case ID", value="CXR-2026-001")
                with gr.Row():
                    patient_age = gr.Textbox(label="Age", value="45")
                    patient_gender = gr.Dropdown(choices=["Male", "Female", "Other"], value="Male", label="Gender")
                symptoms_input = gr.Textbox(
                    label="Clinical Symptoms / Indications",
                    value="Persistent dry cough, low-grade fever, hemoptysis, night sweats"
                )
                
            analyze_btn = gr.Button("🔍 Run Diagnostic Analysis", variant="primary", size="lg")
            
        with gr.Column(scale=6):
            gr.Markdown("### 📊 Diagnostic Classification")
            prediction_output = gr.Label(num_top_classes=3, label="Class Probability Distribution")
            
            gr.Markdown("### 🧠 Explainable AI (XAI) Visual Attributions")
            with gr.Tabs():
                with gr.TabItem("Grad-CAM"):
                    gradcam_output = gr.Image(type="pil", label="Coarse Gradient Activation Heatmap")
                with gr.TabItem("Grad-CAM++"):
                    gradcam_pp_output = gr.Image(type="pil", label="High-Resolution Multi-Instance Localization")
                with gr.TabItem("Guided Backpropagation"):
                    guided_bp_output = gr.Image(type="pil", label="Fine-Grained Micro-Textural Pixel Gradients")
                with gr.TabItem("Saliency Map"):
                    saliency_output = gr.Image(type="pil", label="First-Order Magnitude Saliency Map")
                    
            gr.Markdown("### 📄 Institutional Diagnostic Report")
            pdf_output = gr.File(label="Download Print-Ready Clinical PDF Report")

    analyze_btn.click(
        fn=predict_cxr,
        inputs=[
            image_input,
            backbone_choice,
            cbam_toggle,
            patient_name,
            patient_id,
            patient_age,
            patient_gender,
            symptoms_input
        ],
        outputs=[
            prediction_output,
            gradcam_output,
            gradcam_pp_output,
            guided_bp_output,
            saliency_output,
            pdf_output
        ]
    )
    
    gr.Markdown(
        """
        ---
        *Medical Disclaimer: This application is for academic and clinical research purposes only. AI screening outputs should always be correlated with laboratory microbiology and certified thoracic radiologist consensus.*
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
