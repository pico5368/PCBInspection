"""Anomaly detection inspection using anomalib v2.x.

Uses Engine.predict() for inference (recommended for anomalib >= 2.0).

Requires: pip install pcb-inspection[ml]
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pcb_inspection.common.types import InspectionResult, InspectionType, Severity

logger = logging.getLogger(__name__)


class AnomalyInspector:
    """Anomaly detection using anomalib Engine.predict()."""

    def __init__(self) -> None:
        self._engine = None
        self._model = None
        self._ckpt_path: str | None = None
        self._image_size: tuple[int, int] = (256, 256)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._ckpt_path is not None

    def load(self, model_path: str | None = None, image_size: tuple[int, int] = (256, 256)) -> None:
        """Load an anomalib model checkpoint.

        Args:
            model_path: Path to Lightning .ckpt file.
            image_size: Expected input image size.
        """
        if model_path is None:
            logger.warning("No model path provided for AnomalyInspector")
            return

        path = Path(model_path)
        if not path.exists():
            logger.error("Model path does not exist: %s", model_path)
            return

        self._image_size = image_size

        try:
            from anomalib.engine import Engine

            self._engine = Engine()
            self._model = _detect_and_create_model(path)
            self._ckpt_path = str(path)

            logger.info("Loaded anomaly model from %s", model_path)
        except ImportError:
            logger.error("anomalib not installed. Install with: pip install pcb-inspection[ml]")
        except Exception:
            logger.exception("Failed to load anomaly model from %s", model_path)

    def inspect(
        self,
        roi_image: np.ndarray,
        reference: np.ndarray | None,
        config: dict[str, Any],
    ) -> InspectionResult:
        """Run anomaly detection on a ROI image.

        Config keys:
            component_id: str
            anomaly_threshold: float — score above this = NG (default 0.5)
            warning_threshold: float — score above this = WARNING (default 0.3)
        """
        component_id = config.get("component_id", "unknown")
        ng_threshold = config.get("anomaly_threshold", 0.5)
        warn_threshold = config.get("warning_threshold", 0.3)

        if not self.is_loaded:
            return InspectionResult(
                inspection_type=InspectionType.ANOMALY,
                component_id=component_id,
                severity=Severity.WARNING,
                score=0.5,
                detail="Anomaly model not loaded",
            )

        try:
            # Save image to temp file for Engine.predict
            resized = cv2.resize(roi_image, self._image_size)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, resized)
                temp_path = f.name

            predictions = self._engine.predict(
                model=self._model,
                data_path=temp_path,
                ckpt_path=self._ckpt_path,
                return_predictions=True,
            )

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

            if not predictions:
                raise RuntimeError("No predictions returned")

            pred = predictions[0]

            # Extract score (anomalib normalizes to ~0-1)
            anomaly_score = float(pred.pred_score[0].item()) if pred.pred_score is not None else 0.5
            pred_label = bool(pred.pred_label[0].item()) if pred.pred_label is not None else False
            has_map = pred.anomaly_map is not None

            # Classify
            if anomaly_score > ng_threshold:
                severity = Severity.NG
            elif anomaly_score > warn_threshold:
                severity = Severity.WARNING
            else:
                severity = Severity.OK

            # Our score: 1.0 = normal, 0.0 = anomalous
            score = max(0.0, 1.0 - anomaly_score)

            return InspectionResult(
                inspection_type=InspectionType.ANOMALY,
                component_id=component_id,
                severity=severity,
                score=score,
                detail=f"anomaly_score={anomaly_score:.3f}, pred_label={'anomaly' if pred_label else 'normal'}",
                metadata={
                    "anomaly_score": anomaly_score,
                    "pred_label": pred_label,
                    "has_anomaly_map": has_map,
                },
            )

        except Exception:
            logger.exception("Anomaly inference failed for %s", component_id)
            return InspectionResult(
                inspection_type=InspectionType.ANOMALY,
                component_id=component_id,
                severity=Severity.NG,
                score=0.0,
                detail="Inference error — fail-safe NG",
            )


class MultiModelAnomalyInspector:
    """Manages multiple anomaly models, one per component type."""

    def __init__(self) -> None:
        self._inspectors: dict[str, AnomalyInspector] = {}
        self._default: AnomalyInspector = AnomalyInspector()

    def load_model(
        self, component_type: str, model_path: str, image_size: tuple[int, int] = (256, 256)
    ) -> None:
        inspector = AnomalyInspector()
        inspector.load(model_path, image_size)
        self._inspectors[component_type.upper()] = inspector

    def inspect(
        self,
        roi_image: np.ndarray,
        reference: np.ndarray | None,
        config: dict[str, Any],
    ) -> InspectionResult:
        component_type = config.get("component_type", "").upper()
        inspector = self._inspectors.get(component_type, self._default)
        return inspector.inspect(roi_image, reference, config)

    def load(self, model_path: str | None = None) -> None:
        self._default.load(model_path)


def _detect_and_create_model(ckpt_path: Path):
    """Detect model type from checkpoint and create the model instance."""
    import torch

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Try to detect model class from checkpoint
    model_class_name = None
    if "hyper_parameters" in checkpoint:
        hp = checkpoint["hyper_parameters"]
        model_class_name = hp.get("model", {}).get("class_path", "").split(".")[-1] if isinstance(hp.get("model"), dict) else None

    # Try class name from state dict keys
    if model_class_name is None:
        state_keys = set(checkpoint.get("state_dict", {}).keys())
        if any("memory_bank" in k for k in state_keys):
            model_class_name = "Patchcore"
        elif any("teacher_model" in k or "student_model" in k for k in state_keys):
            model_class_name = "EfficientAd"
        elif any("gaussian" in k for k in state_keys):
            model_class_name = "Padim"

    model_class_name = (model_class_name or "").lower().replace("-", "_")

    if "patchcore" in model_class_name:
        from anomalib.models import Patchcore
        return Patchcore()
    elif "efficient" in model_class_name:
        from anomalib.models import EfficientAd
        return EfficientAd()
    elif "padim" in model_class_name:
        from anomalib.models import Padim
        return Padim()
    else:
        # Default to PatchCore
        logger.warning("Could not detect model type, defaulting to Patchcore")
        from anomalib.models import Patchcore
        return Patchcore()
