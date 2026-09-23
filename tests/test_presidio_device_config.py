"""Regressions for the process-wide Presidio PII device setting.

The real presidio-analyzer and GPUs are optional: a fake AnalyzerEngine checks
that the documented PRESIDIO_DEVICE selector is applied *before* model loading.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_agent.config import SecurityConfig, load_settings
from kb_agent.ingestion.security import PresidioPiiDetector, SecurityScanner


def test_default_config_keeps_presidio_off_gpu() -> None:
    assert SecurityConfig().pii_device == "cpu"
    assert load_settings().security.pii_device == "cpu"


@pytest.mark.parametrize("value, expected", [(" CPU ", "cpu"), ("cuda", "cuda"), (" CUDA:3 ", "cuda:3")])
def test_config_normalizes_valid_devices(value: str, expected: str) -> None:
    assert SecurityConfig(pii_device=value).pii_device == expected


@pytest.mark.parametrize("value", ["", "auto", "cuda:-1", "cuda:abc", "gpu:3", "mps"])
def test_config_rejects_unsupported_device(value: str) -> None:
    with pytest.raises(ValidationError, match="security.pii_device"):
        SecurityConfig(pii_device=value)


def test_nested_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("security:\n  pii_device: cpu\n", encoding="utf-8")
    monkeypatch.setenv("KB_AGENT__SECURITY__PII_DEVICE", "cuda:3")
    assert load_settings(cfg).security.pii_device == "cuda:3"


def test_detector_sets_process_device_before_presidio_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []
    fake = types.ModuleType("presidio_analyzer")

    class AnalyzerEngine:
        def __init__(self) -> None:
            calls.append(os.environ.get("PRESIDIO_DEVICE"))

        def analyze(self, **kwargs):
            return []

    fake.AnalyzerEngine = AnalyzerEngine
    monkeypatch.setitem(sys.modules, "presidio_analyzer", fake)
    monkeypatch.setenv("PRESIDIO_DEVICE", "cuda:0")

    detector = PresidioPiiDetector(device="cpu")
    assert calls == []  # No eager analyzer/model load.
    assert detector.detect("hello", field_path="content") == []
    assert calls == ["cpu"]
    assert detector.detect("again", field_path="content") == []
    assert calls == ["cpu"]  # Analyzer reused.


def test_scanner_forwards_device_and_keeps_injected_detector_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRESIDIO_DEVICE", "cuda:0")
    scanner = SecurityScanner(pii_device="cpu")
    assert isinstance(scanner.pii_detector, PresidioPiiDetector)
    assert scanner.pii_detector.device == "cpu"
    assert os.environ["PRESIDIO_DEVICE"] == "cpu"

    sentinel = object()
    monkeypatch.setenv("PRESIDIO_DEVICE", "cuda:0")
    custom = SecurityScanner(pii_device="cpu", pii_detector=sentinel)
    assert custom.pii_detector is sentinel
    assert os.environ["PRESIDIO_DEVICE"] == "cuda:0"


def test_fastapi_pipeline_uses_configured_pii_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from kb_agent.api import dependencies

    config = load_settings()
    config.security.pii_device = "cpu"
    monkeypatch.setattr(dependencies, "get_settings", lambda: config)
    monkeypatch.setattr(dependencies, "get_raw_store", lambda: object())
    monkeypatch.setattr(dependencies, "get_chunk_indexer", lambda: object())
    monkeypatch.setenv("PRESIDIO_DEVICE", "cuda:0")

    pipeline = dependencies.get_ingestion_pipeline.__wrapped__()
    detector = pipeline.security_scanner.pii_detector
    assert isinstance(detector, PresidioPiiDetector)
    assert detector.device == "cpu"
    assert os.environ["PRESIDIO_DEVICE"] == "cpu"


def test_indexed_cuda_selects_requested_spacy_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """An indexed CUDA device must not silently become spaCy's default GPU0."""
    calls: list[int] = []
    spacy = types.ModuleType("spacy")
    spacy.require_gpu = lambda gpu_id=0: calls.append(gpu_id)
    monkeypatch.setitem(sys.modules, "spacy", spacy)

    presidio = types.ModuleType("presidio_analyzer")
    nlp = types.ModuleType("presidio_analyzer.nlp_engine")

    class SpacyNlpEngine:
        def _enable_gpu(self):
            spacy.require_gpu()  # Reproduce upstream's GPU0 default.

        def load(self):
            self._enable_gpu()

    class NlpEngineProvider:
        def __init__(self, *, nlp_engines):
            self.engines = nlp_engines

        def create_engine(self):
            engine = self.engines[0]()
            engine.load()
            return engine

    class AnalyzerEngine:
        def __init__(self, *, nlp_engine=None):
            assert nlp_engine is not None
            assert os.environ["PRESIDIO_DEVICE"] == "cuda:3"

        def analyze(self, **kwargs):
            return []

    nlp.SpacyNlpEngine = SpacyNlpEngine
    nlp.NlpEngineProvider = NlpEngineProvider
    presidio.AnalyzerEngine = AnalyzerEngine
    monkeypatch.setitem(sys.modules, "presidio_analyzer", presidio)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp)

    detector = PresidioPiiDetector(device="cuda:3")
    assert detector.detect("hello", field_path="content") == []
    assert calls == [3]
