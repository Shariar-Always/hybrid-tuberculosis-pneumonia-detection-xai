# -*- coding: utf-8 -*-
"""
Automated Unit Test Suite for the Hybrid Explainable AI (XAI) Framework
for Tuberculosis & Pneumonia Detection from Chest X-Ray Images.
"""
import sys
import unittest
from pathlib import Path
import torch
import pandas as pd
from PIL import Image

# Add project root to sys.path so `app.*` and `src.*` resolve as packages
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.hybrid_model import ChannelAttention, SpatialAttention, HybridTBPredictor
from app.xai_methods import generate_xai_visualization
from app.report_generator import generate_pdf_report


class TestHybridArchitecture(unittest.TestCase):
    """Tests for CBAM attention blocks and multi-backbone HybridTBPredictor models."""

    def test_cbam_attention_modules(self):
        dummy_features = torch.randn(2, 64, 14, 14)
        ca = ChannelAttention(in_planes=64, ratio=16)
        sa = SpatialAttention(kernel_size=7)

        ca_weights = ca(dummy_features)
        self.assertEqual(ca_weights.shape, (2, 64, 1, 1))

        refined = dummy_features * ca_weights
        sa_weights = sa(refined)
        self.assertEqual(sa_weights.shape, (2, 1, 14, 14))

    def test_densenet121_backbone_forward(self):
        model = HybridTBPredictor(backbone="densenet121", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 3))

    def test_resnet50_backbone_forward(self):
        model = HybridTBPredictor(backbone="resnet50", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 3))

    def test_efficientnet_b0_backbone_forward(self):
        model = HybridTBPredictor(backbone="efficientnet_b0", pretrained=False, use_cbam=True)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 3))

    def test_baseline_backbones_without_cbam_forward(self):
        """Verify all 3 backbones work in Normal / Baseline mode (use_cbam=False)."""
        x = torch.randn(2, 3, 224, 224)
        for backbone in ["densenet121", "resnet50", "efficientnet_b0"]:
            model = HybridTBPredictor(backbone=backbone, pretrained=False, use_cbam=False)
            self.assertFalse(hasattr(model, "channel_attention"))
            self.assertFalse(hasattr(model, "spatial_attention"))
            model.eval()
            with torch.no_grad():
                logits = model(x)
            self.assertEqual(logits.shape, (2, 3))


class TestXAIEngineAndReporting(unittest.TestCase):
    """Tests for the 4 XAI interpretability methods and the ReportLab PDF engine."""

    @classmethod
    def setUpClass(cls):
        cls.model = HybridTBPredictor(backbone="efficientnet_b0", pretrained=False)
        cls.model.eval()
        cls.input_tensor = torch.randn(1, 3, 224, 224)
        cls.dummy_image = Image.new("RGB", (224, 224), color=(120, 120, 120))

    def test_gradcam_visualization(self):
        vis = generate_xai_visualization("Grad-CAM", self.model, self.input_tensor, self.dummy_image)
        self.assertIsInstance(vis, Image.Image)
        self.assertEqual(vis.size, (224, 224))

    def test_gradcam_plusplus_visualization(self):
        vis = generate_xai_visualization("Grad-CAM++", self.model, self.input_tensor, self.dummy_image)
        self.assertIsInstance(vis, Image.Image)
        self.assertEqual(vis.size, (224, 224))

    def test_guided_backpropagation_visualization(self):
        vis = generate_xai_visualization("Guided Backpropagation", self.model, self.input_tensor, self.dummy_image)
        self.assertIsInstance(vis, Image.Image)
        self.assertEqual(vis.size, (224, 224))

    def test_saliency_map_visualization(self):
        vis = generate_xai_visualization("Saliency Map", self.model, self.input_tensor, self.dummy_image)
        self.assertIsInstance(vis, Image.Image)
        self.assertEqual(vis.size, (224, 224))

    def test_clinical_pdf_report_generation(self):
        pdf_bytes = generate_pdf_report(
            patient_name="Test Patient",
            patient_id="PT-1001",
            patient_age="35",
            patient_gender="Male",
            symptoms="Persistent cough and fever",
            prediction="Tuberculosis",
            confidence=98.06,
            insights="- Upper lobe cavitary opacity observed.\n- Recommend sputum AFB correlation.",
            original_image=self.dummy_image,
            xai_overlay=self.dummy_image,
            xai_method_name="Grad-CAM++",
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)


class TestDatasetAndCheckpoints(unittest.TestCase):
    """Tests for dataset metadata integrity and trained model checkpoint availability."""

    def test_metadata_csv_exists_and_valid(self):
        metadata_path = ROOT_DIR / "data" / "metadata.csv"
        self.assertTrue(metadata_path.exists(), "data/metadata.csv should exist")
        df = pd.read_csv(metadata_path)
        expected_cols = {"image_path", "label", "class_name", "split", "source"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        self.assertGreaterEqual(len(df), 13959)
        self.assertEqual(set(df["label"].unique()), {0, 1, 2})

    def test_trained_checkpoints_exist(self):
        models_dir = ROOT_DIR / "models"
        for backbone in ["densenet121", "resnet50", "efficientnet_b0"]:
            ckpt = models_dir / f"best_tb_model_{backbone}.pth"
            self.assertTrue(ckpt.exists(), f"Missing model checkpoint: {ckpt}")


if __name__ == "__main__":
    unittest.main()
