"""Neural detector runtime: lazy, singleton, hash-verified ensemble loader.

Protocol v2.1 Phase 5 production integration. The frozen three-seed neural
detector trained by the private neural final runner is served here as a
runtime tier with explicit fallbacks:

  - ``local-gpu`` / ``cloud-gpu``: full three-seed ensemble (when installed).
  - balanced single-seed: the frozen best seed only.
  - CPU / no-model: the caller falls back to the calibrated logistic tier, then
    the heuristic tier; this module raises :class:`NeuralRuntimeError` and never
    silently degrades.

The manager is a lazy, concurrency-safe singleton. It pins the artifact by
SHA-256 (every checkpoint file is verified against the ensemble manifest before
loading), loads from a local offline cache only, controls device/dtype, and
bounds the window count per document. Heavy dependencies (torch, transformers,
the ``bench.neural`` package) are imported lazily so the backend runs without
them on fixture/heuristic profiles.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from panoptes.analysis.detectors import DetectorAdapter, DetectorScore
from panoptes.schemas import ContentType, OutcomeDistribution, RuntimeProfile

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "models" / "neural"

# The MAGE-trained model is a binary human-vs-machine participation detector;
# it cannot identify the generated/mixed split. We construct the ternary
# monotonically with a documented, unlearned majority-generation prior so that
# ai_generated never exceeds participation. This is a prior, not a measurement.
DEFAULT_MAJORITY_GIVEN_PARTICIPATION = 0.7


class NeuralRuntimeError(RuntimeError):
    """Explicit neural-runtime failure (missing/invalid artifact, hash mismatch)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_neural_stack():
    """Lazily import torch and the bench.neural package; raise explicit error."""
    import sys

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        import torch  # noqa: F401
        from bench.neural.aggregate import aggregate_documents, sigmoid  # noqa: F401
        from bench.neural.model import HierarchicalSummaryHead, WindowEncoder  # noqa: F401
        from bench.neural.windowing import document_windows, pad_windows  # noqa: F401
        from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on environment
        raise NeuralRuntimeError(
            f"neural stack unavailable ({type(exc).__name__}: {exc}); "
            "install the models extra and the pinned neural environment"
        ) from exc
    return {
        "torch": torch,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "aggregate_documents": aggregate_documents,
        "sigmoid": sigmoid,
        "WindowEncoder": WindowEncoder,
        "HierarchicalSummaryHead": HierarchicalSummaryHead,
        "document_windows": document_windows,
        "pad_windows": pad_windows,
    }


def _applicability(
    raw: float,
    seed_probs: list[float],
    seed_window_probs: list,
    x_thresholds,
) -> dict:
    """Applicability/OOD diagnostic for the neural tier (Phase 5).

    A heuristic signal about how far a document sits from the detector's
    declared calibration geometry, combining:

      - ``seed_std``: disagreement among the frozen ensemble seeds.
      - ``segment_std``: disagreement among the document's own window-level
        participation signals (heterogeneous / partially-AI documents score
        high here).
      - ``in_calibration_range``: whether the raw score falls inside the
        isotonic calibrator's fitted range; outside it the calibrator
        extrapolates and the calibrated value is less trustworthy.

    This diagnostic may trigger abstention downstream, but it is NOT proof that
    all distribution shifts are detectable — it is a descriptive signal only.
    """
    import numpy as np

    seed_std = float(np.std(seed_probs)) if len(seed_probs) > 1 else 0.0
    if seed_window_probs:
        mean_window = np.mean(np.stack(seed_window_probs, axis=0), axis=0)
        segment_std = float(np.std(mean_window)) if mean_window.size > 1 else 0.0
    else:
        segment_std = 0.0
    in_range = True
    if x_thresholds:
        lo, hi = float(min(x_thresholds)), float(max(x_thresholds))
        in_range = bool(lo <= raw <= hi)
    flags: list[str] = []
    if not in_range:
        flags.append("out_of_calibration_range")
    if seed_std > 0.15:
        flags.append("ensemble_disagreement")
    if segment_std > 0.30:
        flags.append("segment_disagreement")
    return {
        "seed_std": round(seed_std, 4),
        "segment_std": round(segment_std, 4),
        "in_calibration_range": in_range,
        "flags": flags,
        "note": (
            "Descriptive applicability signal from calibration geometry and "
            "segment/seed disagreement; not proof that all shifts are detectable."
        ),
    }


class NeuralModelManager:
    """Lazy, singleton, concurrency-safe loader for the frozen neural ensemble."""

    _instance: NeuralModelManager | None = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        artifact_dir: str | Path = _DEFAULT_ARTIFACT,
        device: str | None = None,
        max_windows: int = 16,
        max_batch_windows: int = 32,
        single_seed: int | None = None,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.device = device
        self.max_windows = int(max_windows)
        self.max_batch_windows = int(max_batch_windows)
        self.single_seed = single_seed
        self._lock = threading.Lock()
        self._loaded: dict | None = None

    @classmethod
    def instance(cls, artifact_dir: str | Path = _DEFAULT_ARTIFACT, **kwargs) -> NeuralModelManager:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(artifact_dir, **kwargs)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / "ensemble_manifest.json"

    def available(self) -> bool:
        """True iff a well-formed ensemble manifest is present on disk."""
        if not self.manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("schema") == "panoptes-neural-ensemble-v1" and bool(
            manifest.get("seeds")
        )

    def _verify_and_load(self) -> dict:
        stack = _import_neural_stack()
        torch = stack["torch"]
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "panoptes-neural-ensemble-v1":
            raise NeuralRuntimeError("ensemble manifest has an unknown schema")
        winner = manifest["winner"]
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Offline load: rebuild the architecture from the artifact's local config and
        # the tokenizer from the artifact, never re-downloading the base model.
        config_path = self.artifact_dir / "config.json"
        if not config_path.exists():
            raise NeuralRuntimeError(f"missing architecture config {config_path}")
        if manifest.get("config_sha256") and _sha256(config_path) != manifest["config_sha256"]:
            raise NeuralRuntimeError("architecture config hash mismatch")
        config = stack["AutoConfig"].from_pretrained(str(self.artifact_dir))
        tokenizer = stack["AutoTokenizer"].from_pretrained(str(self.artifact_dir))

        seeds = manifest["seeds"]
        if self.single_seed is not None:
            seeds = [s for s in seeds if s["seed"] == self.single_seed] or seeds[:1]
        models = []
        for seed_rec in seeds:
            enc_path = self.artifact_dir / seed_rec["encoder"]
            if not enc_path.exists():
                raise NeuralRuntimeError(f"missing checkpoint {enc_path}")
            if _sha256(enc_path) != seed_rec["encoder_sha256"]:
                raise NeuralRuntimeError(f"checkpoint hash mismatch: {enc_path.name}")
            base = stack["AutoModel"].from_config(config)
            model = stack["WindowEncoder"](encoder=base, hidden_size=int(config.hidden_size))
            from safetensors.torch import load_file

            model.load_state_dict(load_file(str(enc_path)))
            head = None
            if seed_rec.get("summary_head") and seed_rec.get("summary_head_sha256"):
                head_path = self.artifact_dir / seed_rec["summary_head"]
                if not head_path.exists() or _sha256(head_path) != seed_rec["summary_head_sha256"]:
                    raise NeuralRuntimeError(f"summary head hash mismatch: {head_path.name}")
                head = stack["HierarchicalSummaryHead"](hidden=model.hidden_size)
                head.load_state_dict(load_file(str(head_path)))
                head.to(device).eval()
            model.to(device).eval()
            models.append({"model": model, "summary_head": head, "seed": seed_rec["seed"]})
        return {
            "winner": winner,
            "calibration": manifest.get("calibration", {}),
            "models": models,
            "tokenizer": tokenizer,
            "device": device,
            "stack": stack,
        }

    def ensemble(self) -> dict:
        """Return the loaded ensemble, loading it lazily under a lock."""
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    self._loaded = self._verify_and_load()
        return self._loaded

    def score_text(self, text: str) -> dict:
        """Score one document: window -> batched encode -> aggregate -> calibrate.

        Returns a dict with the raw and calibrated participation probability and
        the per-seed probabilities (for transparency / disagreement diagnostics).
        """
        ens = self.ensemble()
        stack = ens["stack"]
        torch = stack["torch"]
        np = __import__("numpy")
        winner = ens["winner"]
        tokenizer = ens["tokenizer"]
        device = ens["device"]

        raw_windows = stack["document_windows"](
            text,
            tokenizer,
            max_length=winner["max_length"],
            overlap=winner["overlap"],
            max_windows=self.max_windows,
        )
        if not raw_windows:
            raise NeuralRuntimeError("document produced no windows")
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )
        wins = stack["pad_windows"](raw_windows, pad_id=pad_id, max_length=winner["max_length"])
        input_ids = np.array([w.input_ids for w in wins], dtype=np.int64)
        attention_mask = np.array([w.attention_mask for w in wins], dtype=np.int64)
        spans = [(w.token_start, w.token_end) for w in raw_windows]
        n_tokens = max((w.token_end for w in raw_windows), default=0)

        seed_probs: list[float] = []
        seed_window_probs: list[np.ndarray] = []
        with torch.no_grad():
            for rec in ens["models"]:
                model = rec["model"]
                head = rec["summary_head"]
                logits_out = []
                cls_out = []
                for start in range(0, len(input_ids), self.max_batch_windows):
                    ids = torch.from_numpy(input_ids[start : start + self.max_batch_windows]).to(
                        device
                    )
                    mask = torch.from_numpy(
                        attention_mask[start : start + self.max_batch_windows]
                    ).to(device)
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                    ):
                        logits, cls = model(ids, mask)
                    logits_out.append(logits.float().cpu().numpy())
                    if head is not None:
                        cls_out.append(cls.float().cpu().numpy())
                window_logits = np.concatenate(logits_out, axis=0)
                # Per-window participation signal (segment disagreement diagnostic).
                seed_window_probs.append(
                    stack["sigmoid"](window_logits[:, 1] - window_logits[:, 0]).astype(np.float64)
                )
                if head is not None:
                    # Hierarchical summary head over this document's window embeddings.
                    window_embeds = np.concatenate(cls_out, axis=0)  # [n_windows, hidden]
                    w = min(len(window_embeds), self.max_windows)
                    embeds = np.zeros((1, w, window_embeds.shape[1]), dtype=np.float32)
                    head_mask = np.zeros((1, w), dtype=bool)
                    embeds[0, :w] = window_embeds[:w]
                    head_mask[0, :w] = True
                    head_logits = head(
                        torch.from_numpy(embeds).to(device), torch.from_numpy(head_mask).to(device)
                    )
                    logit_diff = float((head_logits[:, 1] - head_logits[:, 0])[0])
                    prob = float(stack["sigmoid"](logit_diff))
                else:
                    prob = float(
                        stack["aggregate_documents"]([window_logits], [spans], [n_tokens])[0]
                    )
                seed_probs.append(prob)

        raw = float(np.mean(seed_probs))
        calibrator = (ens["calibration"] or {}).get("binary_calibrator") or {}
        xs = calibrator.get("x_thresholds")
        ys = calibrator.get("y_thresholds")
        calibrated = float(np.interp(raw, xs, ys)) if xs and ys else raw
        applicability = _applicability(raw, seed_probs, seed_window_probs, xs)
        return {
            "raw_participation": raw,
            "calibrated_participation": min(max(calibrated, 0.0), 1.0),
            "seed_probabilities": seed_probs,
            "n_windows": len(input_ids),
            "n_seeds": len(ens["models"]),
            "applicability": applicability,
        }


class NeuralProseDetector(DetectorAdapter):
    """Frozen neural ensemble as a DetectorAdapter, with explicit errors."""

    id = "neural-ensemble-v1"
    min_tokens = 50
    content_types = (ContentType.PROSE, ContentType.MIXED)
    languages = ("en",)

    def __init__(self, manager: NeuralModelManager):
        self.manager = manager

    def score(self, text: str, content_type: ContentType, language: str) -> DetectorScore:
        if language != "en":
            return self._abstain("Neural detector is calibrated only for English prose.")
        result = self.manager.score_text(text)
        participation = result["calibrated_participation"]
        # Monotonic ternary: ai_generated <= participation; the generated/mixed
        # split uses a documented unlearned prior (MAGE is binary).
        majority = DEFAULT_MAJORITY_GIVEN_PARTICIPATION
        ai_generated = participation * majority
        ai_refined = participation * (1.0 - majority)
        human = max(0.0, 1.0 - participation)
        distribution = OutcomeDistribution(
            human=human, ai_generated=ai_generated, ai_refined_or_mixed=ai_refined
        ).normalized()
        return DetectorScore(
            distribution=distribution,
            raw_score=participation,
            detector_id=self.id,
        )

    def _abstain(self, reason: str) -> DetectorScore:
        return DetectorScore(
            distribution=OutcomeDistribution(
                human=1 / 3, ai_generated=1 / 3, ai_refined_or_mixed=1 / 3
            ),
            raw_score=0.5,
            detector_id=self.id,
            abstain_reason=reason,
        )


def try_neural_detector(settings) -> NeuralProseDetector | None:
    """Return the neural detector for the document tier, or ``None`` to fall back.

    The neural ensemble is the primary statistical detector only where it is
    practical and calibrated: a GPU runtime profile with the frozen artifact
    installed. Fixture mode never uses it, and CPU profiles fall back to the
    calibrated logistic / heuristic tiers (a three-seed transformer ensemble is
    not interactive on CPU). A missing or malformed artifact also falls back —
    the manager raises :class:`NeuralRuntimeError` only on an explicit load of a
    corrupt artifact, never on the availability check here.
    """
    if not getattr(settings, "neural_enabled", True):
        return None
    profile = getattr(settings, "profile", RuntimeProfile.FIXTURE)
    if profile not in {RuntimeProfile.LOCAL_GPU, RuntimeProfile.CLOUD_GPU}:
        return None
    artifact_dir = getattr(settings, "neural_artifact_dir", str(_DEFAULT_ARTIFACT))
    manager = NeuralModelManager.instance(artifact_dir=artifact_dir)
    if not manager.available():
        return None
    return NeuralProseDetector(manager)
