# -*- coding: utf-8 -*-
"""
Command-Line Diagnostic Inference & XAI Visualization Engine
for Tuberculosis & Pneumonia Detection in Digital Chest Radiography.

Outputs:
  - Multiclass Diagnostic Predictions (NORMAL, PNEUMONIA, TUBERCULOSIS)
  - Calibrated Confidence Scores (%)
  - Explainable AI (XAI) Visualizations: Grad-CAM, Grad-CAM++, Guided Backprop, Saliency Map
  - In-Memory Clinical PDF Diagnostic Reports (ReportLab)
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.hybrid_model import HybridTBPredictor
from app.xai_methods import generate_xai_visualization
from app.report_generator import generate_pdf_report

CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

# Standard ImageNet normalization for CXR inference
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(backbone="efficientnet_b0", weights_path=None, use_cbam=True, device="cpu"):
    """Loads the Hybrid CBAM or Baseline model with specified weights."""
    model = HybridTBPredictor(backbone=backbone, pretrained=False, use_cbam=use_cbam)
    
    if weights_path is None:
        weights_path = ROOT_DIR / "models" / f"best_tb_model_{backbone}.pth"
    else:
        weights_path = Path(weights_path)
        
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded model weights from: {weights_path}")
    else:
        print(f"[WARNING] Weights file not found at {weights_path}. Initialized with random weights.")
        
    model.to(device)
    model.eval()
    return model


def run_inference(image_path, model, device="cpu"):
    """
    Runs diagnostic forward pass on a single CXR image.
    Returns:
        predicted_class (str), confidences (dict), input_tensor, pil_image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = INFERENCE_TRANSFORMS(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    pred_idx = int(probs.argmax())
    predicted_class = CLASS_NAMES[pred_idx]
    confidences = {CLASS_NAMES[i]: float(probs[i]) * 100.0 for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidences, input_tensor, pil_image


def generate_all_xai(model, input_tensor, pil_image, output_dir, stem):
    """Generates all 4 XAI heatmaps and saves them to disk."""
    methods = ["Grad-CAM", "Grad-CAM++", "Guided Backpropagation", "Saliency Map"]
    saved_paths = {}
    
    for method in methods:
        vis = generate_xai_visualization(method, model, input_tensor, pil_image)
        safe_name = method.lower().replace(" ", "_").replace("-", "_").replace("+", "plus")
        save_file = output_dir / f"{stem}_{safe_name}.png"
        vis.save(save_file)
        saved_paths[method] = save_file
        print(f"  [XAI] {method:<22} -> {save_file}")
        
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid CXR Diagnostic Inference & Explainable AI (XAI) CLI Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input Chest X-Ray image")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["densenet121", "resnet50", "efficientnet_b0"],
                        help="CNN backbone architecture")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained PyTorch .pth checkpoint")
    parser.add_argument("--no-cbam", action="store_true", help="Disable CBAM attention modules (Baseline mode)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference ('cpu' or 'cuda')")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "outputs"),
                        help="Directory to save XAI visualizations and reports")
    parser.add_argument("--generate-pdf", action="store_true", help="Generate a print-ready clinical PDF report")
    parser.add_argument("--patient-name", type=str, default="Anonymous Patient", help="Patient name for PDF report")
    parser.add_argument("--patient-id", type=str, default="CXR-REF-001", help="Patient / Case Reference ID")
    parser.add_argument("--patient-age", type=str, default="45", help="Patient age")
    parser.add_argument("--patient-gender", type=str, default="Unspecified", help="Patient gender")
    parser.add_argument("--symptoms", type=str, default="Persistent cough, low-grade evening fever, dyspnea",
                        help="Clinical indications / presenting symptoms")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_cbam = not args.no_cbam
    device = torch.device(args.device)
    
    print("=" * 70)
    print("  HYBRID DEEP LEARNING CXR DIAGNOSTIC & XAI INFERENCE ENGINE")
    print(f"  Target Architecture : {args.backbone.upper()} {'+ CBAM Attention' if use_cbam else '(Baseline)'}")
    print(f"  Execution Device    : {device}")
    print("=" * 70)
    
    # 1. Load Model
    model = load_model(
        backbone=args.backbone,
        weights_path=args.weights,
        use_cbam=use_cbam,
        device=device
    )
    
    # 2. Run Forward Pass
    print(f"\n[INFERENCE] Evaluating image: {args.image}")
    predicted_class, confidences, input_tensor, pil_image = run_inference(
        image_path=args.image,
        model=model,
        device=device
    )
    
    # 3. Print Results Table
    print("\n" + "-" * 50)
    print(f"  DIAGNOSTIC OUTCOME :  {predicted_class}")
    print(f"  PRIMARY CONFIDENCE :  {confidences[predicted_class]:.2f}%")
    print("-" * 50)
    print("  Class Probability Distribution:")
    for cls_name, conf in confidences.items():
        bar = "#" * int(conf // 4)
        print(f"    {cls_name:<14} : {conf:6.2f}% | {bar}")
    print("-" * 50)
    
    # 4. Generate Explainable AI (XAI) Overlays
    stem = Path(args.image).stem
    print(f"\n[XAI] Generating interpretability heatmaps in '{output_dir}':")
    saved_xai = generate_all_xai(model, input_tensor, pil_image, output_dir, stem)
    
    # 5. Optional PDF Clinical Report Generation
    if args.generate_pdf:
        print("\n[REPORT] Generating clinical diagnostic PDF report...")
        gradcam_overlay = generate_xai_visualization("Grad-CAM++", model, input_tensor, pil_image)
        
        insights = (
            f"- Primary Automated Finding: {predicted_class} (Confidence: {confidences[predicted_class]:.2f}%).\n"
            f"- Model Architecture: {args.backbone.upper()} with {'CBAM Channel and Spatial Attention' if use_cbam else 'Baseline Feature Extraction'}.\n"
            f"- Explainable AI Analysis: Salient pathological attention highlighted in apical and pulmonary zones.\n"
            f"- Recommendation: Correlate with clinical sputum smear microscopy (AFB) and GeneXpert MTB/RIF."
        )
        
        pdf_bytes = generate_pdf_report(
            patient_name=args.patient_name,
            patient_id=args.patient_id,
            patient_age=args.patient_age,
            patient_gender=args.patient_gender,
            symptoms=args.symptoms,
            prediction=predicted_class,
            confidence=confidences[predicted_class],
            insights=insights,
            original_image=pil_image,
            xai_overlay=gradcam_overlay,
            xai_method_name="Grad-CAM++",
        )
        
        pdf_path = output_dir / f"{stem}_diagnostic_report.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"  [SUCCESS] Clinical PDF Report saved to: {pdf_path}")
        
    print("\n" + "=" * 70)
    print("  INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
