"""Application state management and dependency injection for InspectAI backend."""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import Request
import torch

from src.explainability.explainer import VibrationExplainer, DEFAULT_CLASS_NAMES
from src.inference.domain_adapter import DomainAdapter
from src.inference.predictor import StreamingPredictor
from src.models.cnn_1d import VibrationCNN1D
from src.preprocessing.signal_cleaner import SignalCleaner
from src.training.normalizer import VibrationNormalizer
from src.utils.config_parser import get_resolved_paths, load_config

logger = logging.getLogger("InspectAI.Backend")


@dataclass
class AppState:
    """Singleton container maintaining loaded ML models and preprocessors in memory."""

    model: Optional[VibrationCNN1D] = None
    normalizer: Optional[VibrationNormalizer] = None
    signal_cleaner: Optional[SignalCleaner] = None
    predictor: Optional[StreamingPredictor] = None
    explainer: Optional[VibrationExplainer] = None
    domain_adapter: Optional[DomainAdapter] = None
    class_names: List[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    device: str = "cpu"
    is_loaded: bool = False
    load_error: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


def load_model_artifacts(state: AppState) -> None:
    """Load model checkpoint, normalizer, and signal cleaner into application state once on startup.

    Raises:
        FileNotFoundError: If checkpoint or normalizer files are missing.
        RuntimeError: If checkpoint loading or model instantiation fails.
    """
    logger.info("Initializing InspectAI inference artifacts on CPU...")
    try:
        config = load_config()
        paths = get_resolved_paths(config)
        state.config = config

        # Support optional dynamic loading of final candidate model from models/final/
        model_dir_env = os.getenv("INSPECTAI_MODEL_DIR", "").strip()
        use_final = os.getenv("INSPECTAI_USE_FINAL_MODEL", "").lower() in ("1", "true")
        if use_final or (model_dir_env and Path(model_dir_env).resolve() == Path("models/final").resolve()):
            final_dir = Path("models/final")
            logger.info(f"Loading candidate release model from {final_dir} via StreamingPredictor.from_final_model...")
            predictor = StreamingPredictor.from_final_model(model_dir=final_dir, device="cpu")
            explainer = VibrationExplainer(
                model=predictor.model,
                normalizer=predictor.normalizer,
                spectral_normalizer=predictor.spectral_normalizer,
                class_names=predictor.class_names,
                device="cpu",
            )
            state.model = predictor.model
            state.normalizer = predictor.normalizer
            state.signal_cleaner = SignalCleaner.from_config(config)
            state.predictor = predictor
            state.explainer = explainer
            state.class_names = predictor.class_names
            state.device = "cpu"
            state.is_loaded = True
            state.load_error = None

            cwru_path = Path("data/processed/cwru_segmented_windows.npz")
            if cwru_path.is_file():
                try:
                    adapter = DomainAdapter.from_processed_cwru(cwru_path, predictor=predictor)
                    state.domain_adapter = adapter
                    predictor.domain_adapter = adapter
                    logger.info("Candidate release model and domain adapter successfully initialized.")
                except Exception as exc:
                    logger.warning(f"Domain adapter reference init deferred: {exc}")

            logger.info("Candidate release model (SpectralFusionCNN) successfully loaded into backend state.")
            return

        ckpt_path = paths["model_checkpoint_dir"] / "inspectai_final_cnn.pth"
        norm_path = paths["model_checkpoint_dir"] / "inspectai_final_normalizer.json"

        if not ckpt_path.is_file():
            err_msg = f"Final model checkpoint not found at: {ckpt_path}"
            logger.error(err_msg)
            state.load_error = err_msg
            state.is_loaded = False
            raise FileNotFoundError(err_msg)

        if not norm_path.is_file():
            err_msg = f"Final normalizer file not found at: {norm_path}"
            logger.error(err_msg)
            state.load_error = err_msg
            state.is_loaded = False
            raise FileNotFoundError(err_msg)

        # 1. Load trained 1D CNN
        logger.info(f"Loading checkpoint from: {ckpt_path.name}")
        model = VibrationCNN1D.from_config(config)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # 2. Load fitted normalizer
        logger.info(f"Loading normalizer from: {norm_path.name}")
        normalizer = VibrationNormalizer.load(norm_path)

        # 3. Instantiate signal cleaner
        cleaner = SignalCleaner.from_config(config)

        # 4. Instantiate predictor and explainer
        class_names = list(DEFAULT_CLASS_NAMES)
        predictor = StreamingPredictor(
            model=model,
            class_names=class_names,
            normalizer=normalizer,
            device="cpu",
        )
        explainer = VibrationExplainer(
            model=model,
            normalizer=normalizer,
            class_names=class_names,
            device="cpu",
        )

        state.model = model
        state.normalizer = normalizer
        state.signal_cleaner = cleaner
        state.predictor = predictor
        state.explainer = explainer
        state.class_names = class_names
        state.device = "cpu"
        state.is_loaded = True
        state.load_error = None

        logger.info("InspectAI backend artifacts successfully loaded and ready for inference.")

    except Exception as exc:
        state.is_loaded = False
        state.load_error = str(exc)
        logger.exception(f"Fatal error initializing model backend artifacts: {exc}")
        raise


def get_app_state(request: Request) -> AppState:
    """Dependency provider returning the singleton application state."""
    return request.app.state.app_state

