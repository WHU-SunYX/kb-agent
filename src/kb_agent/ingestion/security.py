"""Security scanning and sanitization for normalized knowledge documents.

This module is the security boundary between source parsing and downstream
chunking/indexing.  It intentionally works on :class:`NormalizedDocument`
objects so ChatGPT exports, PDF/DOCX documents, and future source adapters all
share one policy implementation.

The scanner delegates detection to established open-source components:

* ``detect-secrets`` for credentials/tokens/secrets.
* Microsoft Presidio Analyzer for PII detection.

kb-agent only owns policy orchestration, safe finding metadata, and optional
redaction of normalized text.  Raw evidence in ``RawStore`` is never modified.

Important behavior
------------------
* Secrets are BLOCKed by default.  Their raw value is never copied into a
  ``SecurityFinding``.
* PII is MARKed by default; policy can change it to REDACT or BLOCK.
* Internal/private IP addresses and local filesystem paths are MARKed as
  contextual sensitive data rather than deleted by default.
* Nested ``document.metadata`` and ``source.metadata`` text is scanned too, but
  operational fields such as filenames, hashes, MIME types, IDs, and conversion
  bookkeeping are skipped by default.  This keeps structured conversation text
  covered without treating kb-agent's own provenance metadata as user content.
* Entropy-only secret findings require credential-like context by default.  This
  avoids blocking ordinary prose, hashes, model identifiers, and similar text.
* Exact duplicate findings are collapsed before policy decisions are returned.
* If redaction changes ``document.content``, ``content_hash`` is recomputed.

Dependencies are lazy-loaded so importing kb-agent does not require security
extras.  Install them with ``pip install -e '.[dev,security]'`` before using the
default secret/PII detectors.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import ipaddress
import os
import re
from typing import Any, Iterable, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kb_agent.models import NormalizedDocument, SecurityMetadata


class SecurityDependencyError(RuntimeError):
    """Raised when an enabled optional security detector is unavailable."""


class SecurityAction(str, Enum):
    """Policy action assigned to one security finding."""

    ALLOW = "allow"
    MARK = "mark"
    REDACT = "redact"
    BLOCK = "block"


class SecurityFindingKind(str, Enum):
    """Broad category of a security finding."""

    SECRET = "secret"
    PII = "pii"
    CONTEXT = "context"


_ACTION_PRIORITY = {
    SecurityAction.ALLOW: 0,
    SecurityAction.MARK: 1,
    SecurityAction.REDACT: 2,
    SecurityAction.BLOCK: 3,
}


class _SecurityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SecurityFinding(_SecurityModel):
    """Safe metadata about a detected sensitive item.

    The matched value itself is deliberately excluded.  ``evidence_hash`` may
    contain a one-way hash supplied by a detector and is useful for correlating
    repeated findings without exposing the secret/PII value.
    """

    kind: SecurityFindingKind
    category: str = Field(min_length=1)
    detector: str = Field(min_length=1)
    action: SecurityAction
    field_path: str = Field(min_length=1)

    entity_type: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    line_number: int | None = Field(default=None, ge=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    evidence_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("end")
    @classmethod
    def _validate_end(cls, value: int | None, info):
        start = info.data.get("start")
        if value is not None and start is not None and value < start:
            raise ValueError("end must be greater than or equal to start")
        return value


class SecurityPolicy(_SecurityModel):
    """Policy applied to findings from each detector family."""

    secret_action: SecurityAction = SecurityAction.BLOCK
    pii_action: SecurityAction = SecurityAction.MARK
    context_action: SecurityAction = SecurityAction.MARK

    scan_title: bool = True
    scan_author: bool = True
    scan_metadata: bool = True
    scan_source_metadata: bool = True
    scan_operational_metadata: bool = False
    entropy_secrets_require_context: bool = True

    @field_validator("secret_action")
    @classmethod
    def _secret_action_must_not_redact(cls, value: SecurityAction) -> SecurityAction:
        # detect-secrets reliably identifies the type/line but does not expose a
        # stable public span API.  Blocking/marking is therefore safer than
        # attempting partial credential redaction.
        if value is SecurityAction.REDACT:
            raise ValueError("secret_action cannot be 'redact'; use 'block' or 'mark'")
        return value


class SecurityScanResult(_SecurityModel):
    """Result of scanning one normalized document."""

    document_id: str = Field(min_length=1)
    decision: SecurityAction
    findings: list[SecurityFinding] = Field(default_factory=list)
    security: SecurityMetadata
    document: NormalizedDocument | None = None

    @property
    def blocked(self) -> bool:
        return self.decision is SecurityAction.BLOCK


@dataclass(frozen=True)
class DetectedItem:
    """Detector-neutral internal match representation."""

    kind: SecurityFindingKind
    category: str
    detector: str
    field_path: str
    entity_type: str | None = None
    confidence: float | None = None
    line_number: int | None = None
    start: int | None = None
    end: int | None = None
    evidence_hash: str | None = None
    metadata: dict[str, Any] | None = None


class SecurityDetector(Protocol):
    """Protocol implemented by secret/PII/context detector adapters."""

    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        ...


_ENTROPY_SECRET_TYPES = frozenset(
    {
        "base64 high entropy string",
        "hex high entropy string",
    }
)

_CREDENTIAL_CONTEXT_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"\b(?:api[_-]?key|access[_-]?key|secret|token|password|passwd|pwd|"
    r"credential|client[_-]?secret|private[_-]?key|authorization)\b"
    r"\s*(?:=|:)\s*['\"]?\S{6,}"
    r"|\bbearer\s+\S{8,}"
    r")"
)

# These fields are generated or structural kb-agent provenance, not user text.
# Scanning them creates deterministic false positives (for example SHA-256
# values triggering HexHighEntropyString, or ``test.md`` being treated as a URL).
_OPERATIONAL_METADATA_KEYS = frozenset(
    {
        "content_hash",
        "conversion_status",
        "created_at",
        "current_node",
        "docling_document_name",
        "document_id",
        "extension",
        "filename",
        "format",
        "hash",
        "id",
        "message_id",
        "mime_type",
        "parent_id",
        "provider",
        "raw_sha256",
        "raw_size_bytes",
        "raw_uri",
        "sha256",
        "source_id",
        "source_type",
        "updated_at",
    }
)


class DetectSecretsDetector:
    """Thin adapter around Yelp's ``detect-secrets`` scanner.

    ``detect-secrets`` includes entropy-based plugins which are useful for source
    code/configuration but too aggressive for natural-language knowledge bases.
    By default, entropy-only matches are retained only when their line contains
    credential-like context (for example ``api_key=...`` or ``Bearer ...``).
    High-confidence typed detectors remain unaffected.
    """

    name = "detect-secrets"

    def __init__(self, *, require_context_for_entropy: bool = True) -> None:
        self.require_context_for_entropy = require_context_for_entropy

    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        if not text:
            return []

        try:
            from detect_secrets.core.scan import scan_line
            from detect_secrets.settings import default_settings
        except ImportError as exc:
            raise SecurityDependencyError(
                "Secret scanning requires detect-secrets. Install the security extra: "
                "pip install -e '.[dev,security]'"
            ) from exc

        findings: list[DetectedItem] = []
        with default_settings():
            for line_number, line in enumerate(text.splitlines(), start=1):
                for secret in scan_line(line):
                    secret_type = str(getattr(secret, "type", "secret")).strip() or "secret"
                    if (
                        self.require_context_for_entropy
                        and _is_entropy_secret_type(secret_type)
                        and not _has_credential_context(line)
                    ):
                        continue
                    secret_hash = getattr(secret, "secret_hash", None)
                    findings.append(
                        DetectedItem(
                            kind=SecurityFindingKind.SECRET,
                            category=f"secret:{_slug(secret_type)}",
                            detector=self.name,
                            field_path=field_path,
                            entity_type=secret_type,
                            line_number=line_number,
                            evidence_hash=str(secret_hash) if secret_hash else None,
                        )
                    )
        return findings


class PresidioPiiDetector:
    """Thin adapter around Microsoft Presidio Analyzer.

    ``analyzer`` can be injected for tests or custom multilingual deployments.
    When omitted, Presidio's default ``AnalyzerEngine`` is created lazily.
    ``device`` selects Presidio's process-wide ``PRESIDIO_DEVICE`` override
    before any analyzer is loaded; changing it requires a process restart.
    """

    name = "presidio"

    def __init__(
        self,
        *,
        analyzer: Any | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
        entities: Sequence[str] | None = None,
        device: str | None = None,
    ) -> None:
        self._analyzer = analyzer
        self.language = language
        self.score_threshold = score_threshold
        self.entities = list(entities) if entities is not None else None
        self.device = device
        if device is not None:
            # Presidio AnalyzerEngine does not accept a device= constructor
            # argument. Its supported override is a process environment variable.
            # Set it before the lazy import/NLP load (and any lazy recognizers).
            normalized = device.strip().lower()
            if not re.fullmatch(r"cpu|cuda(?::(?:0|[1-9][0-9]*))?", normalized):
                raise ValueError("Presidio device must be 'cpu', 'cuda' or 'cuda:N'")
            self.device = normalized
            if analyzer is None:
                os.environ["PRESIDIO_DEVICE"] = normalized

    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        if not text:
            return []

        analyzer = self._get_analyzer()
        kwargs: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "score_threshold": self.score_threshold,
        }
        if self.entities is not None:
            kwargs["entities"] = self.entities

        try:
            results = analyzer.analyze(**kwargs)
        except Exception as exc:
            raise SecurityDependencyError(
                "Presidio PII analysis failed. Ensure presidio-analyzer and an NLP engine/model "
                "for the configured language are installed. For the default English setup, "
                "Presidio commonly uses a spaCy model such as en_core_web_lg."
            ) from exc

        findings: list[DetectedItem] = []
        for result in results:
            entity_type = str(getattr(result, "entity_type", "PII")).strip() or "PII"
            start = _optional_int(getattr(result, "start", None))
            end = _optional_int(getattr(result, "end", None))
            score = _optional_float(getattr(result, "score", None))
            if start is None or end is None or start < 0 or end < start or end > len(text):
                # Do not trust malformed spans from custom recognizers.
                continue
            matched = text[start:end]
            findings.append(
                DetectedItem(
                    kind=SecurityFindingKind.PII,
                    category=f"pii:{_slug(entity_type)}",
                    detector=self.name,
                    field_path=field_path,
                    entity_type=entity_type,
                    confidence=score,
                    start=start,
                    end=end,
                    evidence_hash=sha256(matched.encode("utf-8")).hexdigest(),
                )
            )
        return findings

    def _get_analyzer(self) -> Any:
        if self._analyzer is not None:
            return self._analyzer
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            raise SecurityDependencyError(
                "PII scanning requires presidio-analyzer. Install the security extra: "
                "pip install -e '.[dev,security]'"
            ) from exc
        try:
            if self.device is not None and self.device.startswith("cuda:"):
                # Presidio's stock SpacyNlpEngine currently calls
                # spacy.require_gpu() without gpu_id, which selects GPU0 even
                # when PRESIDIO_DEVICE=cuda:N. Override only its device hook,
                # retaining Presidio's default spaCy model/NER configuration.
                from presidio_analyzer.nlp_engine import NlpEngineProvider, SpacyNlpEngine

                gpu_id = int(self.device.split(":", 1)[1])

                class _IndexedSpacyEngine(SpacyNlpEngine):
                    def _enable_gpu(self) -> None:
                        import spacy

                        # Fail rather than silently moving PII processing to
                        # GPU0 or falling back to an unexpected device.
                        spacy.require_gpu(gpu_id=gpu_id)

                engine = NlpEngineProvider(nlp_engines=(_IndexedSpacyEngine,)).create_engine()
                self._analyzer = AnalyzerEngine(nlp_engine=engine)
            else:
                self._analyzer = AnalyzerEngine()
        except Exception as exc:
            raise SecurityDependencyError(
                f"Unable to initialize Presidio AnalyzerEngine for device "
                f"{self.device or 'default'!r}. Check the NLP model, "
                "Presidio version and GPU dependencies."
            ) from exc
        return self._analyzer


_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_UNIX_PATH_RE = re.compile(
    r"(?<![\w])/(?:home|root|var|etc|opt|srv|mnt|workspace|dev|tmp)(?:/[^\s\"'<>`\[\](){}]+)+"
)
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s<>:\"|?*]+\\?)+")


class ContextMetadataDetector:
    """Mark internal infrastructure context without treating it as a secret.

    This intentionally handles only high-confidence forms useful to the AI-SSD
    use case: private IP addresses and common absolute local paths.  It does not
    try to replace Presidio or a DLP product.
    """

    name = "kb-agent-context"

    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        findings: list[DetectedItem] = []

        for match in _IPV4_RE.finditer(text):
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if not address.is_private:
                continue
            findings.append(
                DetectedItem(
                    kind=SecurityFindingKind.CONTEXT,
                    category="internal_ip",
                    detector=self.name,
                    field_path=field_path,
                    entity_type="PRIVATE_IP",
                    start=match.start(),
                    end=match.end(),
                    evidence_hash=sha256(match.group(0).encode("utf-8")).hexdigest(),
                )
            )

        for regex, entity_type in (
            (_UNIX_PATH_RE, "LOCAL_UNIX_PATH"),
            (_WINDOWS_PATH_RE, "LOCAL_WINDOWS_PATH"),
        ):
            for match in regex.finditer(text):
                findings.append(
                    DetectedItem(
                        kind=SecurityFindingKind.CONTEXT,
                        category="local_path",
                        detector=self.name,
                        field_path=field_path,
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        evidence_hash=sha256(match.group(0).encode("utf-8")).hexdigest(),
                    )
                )

        return findings


class SecurityScanner:
    """Scan and optionally sanitize normalized documents before indexing."""

    def __init__(
        self,
        *,
        policy: SecurityPolicy | None = None,
        secret_detector: SecurityDetector | None = None,
        pii_detector: SecurityDetector | None = None,
        pii_device: str | None = None,
        context_detector: SecurityDetector | None = None,
        enable_secret_scan: bool = True,
        enable_pii_scan: bool = True,
        enable_context_scan: bool = True,
    ) -> None:
        self.policy = policy or SecurityPolicy()
        self.secret_detector = secret_detector or DetectSecretsDetector(
            require_context_for_entropy=self.policy.entropy_secrets_require_context
        )
        self.pii_detector = (
            pii_detector if pii_detector is not None else PresidioPiiDetector(device=pii_device)
        )
        self.context_detector = context_detector or ContextMetadataDetector()
        self.enable_secret_scan = enable_secret_scan
        self.enable_pii_scan = enable_pii_scan
        self.enable_context_scan = enable_context_scan

    def scan(self, document: NormalizedDocument) -> SecurityScanResult:
        """Scan ``document`` and return an indexing-safe result.

        A BLOCK decision returns ``document=None``.  Otherwise a deep-copied
        document is returned, with security metadata updated and any REDACT
        policy applied to normalized text fields.
        """

        payload = document.model_dump(mode="python")
        fields = list(self._iter_scannable_fields(payload))
        findings: list[SecurityFinding] = []
        redactions: dict[tuple[Any, ...], list[tuple[int, int, str]]] = {}

        for path, text in fields:
            field_path = _format_path(path)
            detected: list[DetectedItem] = []
            if self.enable_secret_scan:
                detected.extend(self.secret_detector.detect(text, field_path=field_path))
            if self.enable_pii_scan:
                detected.extend(self.pii_detector.detect(text, field_path=field_path))
            if self.enable_context_scan:
                detected.extend(self.context_detector.detect(text, field_path=field_path))

            detected = _deduplicate_detected_items(detected)
            for item in detected:
                action = self._action_for(item.kind)
                finding = SecurityFinding(
                    kind=item.kind,
                    category=item.category,
                    detector=item.detector,
                    action=action,
                    field_path=item.field_path,
                    entity_type=item.entity_type,
                    confidence=item.confidence,
                    line_number=item.line_number,
                    start=item.start,
                    end=item.end,
                    evidence_hash=item.evidence_hash,
                    metadata=item.metadata or {},
                )
                findings.append(finding)

                if (
                    action is SecurityAction.REDACT
                    and item.start is not None
                    and item.end is not None
                ):
                    redactions.setdefault(path, []).append(
                        (item.start, item.end, _replacement_for(item))
                    )

        findings = _deduplicate_findings(findings)
        decision = _highest_action(finding.action for finding in findings)
        security = self._updated_security(document.security, findings)

        if decision is SecurityAction.BLOCK:
            return SecurityScanResult(
                document_id=document.document_id,
                decision=decision,
                findings=findings,
                security=security,
                document=None,
            )

        for path, spans in redactions.items():
            current = _get_path(payload, path)
            if isinstance(current, str):
                _set_path(payload, path, _apply_redactions(current, spans))

        payload["security"] = security.model_dump(mode="python")
        metadata = payload.setdefault("metadata", {})
        metadata["security_scan"] = _security_summary(decision, findings)

        if redactions:
            payload["content_hash"] = sha256(payload["content"].encode("utf-8")).hexdigest()

        sanitized_document = NormalizedDocument.model_validate(payload)
        return SecurityScanResult(
            document_id=document.document_id,
            decision=decision,
            findings=findings,
            security=security,
            document=sanitized_document,
        )

    def _action_for(self, kind: SecurityFindingKind) -> SecurityAction:
        if kind is SecurityFindingKind.SECRET:
            return self.policy.secret_action
        if kind is SecurityFindingKind.PII:
            return self.policy.pii_action
        return self.policy.context_action

    def _iter_scannable_fields(
        self, payload: dict[str, Any]
    ) -> Iterable[tuple[tuple[Any, ...], str]]:
        yield ("content",), payload["content"]

        if self.policy.scan_title and payload.get("title"):
            yield ("title",), payload["title"]
        if self.policy.scan_author and payload.get("author"):
            yield ("author",), payload["author"]
        if self.policy.scan_metadata:
            yield from self._iter_metadata_fields(
                payload.get("metadata", {}),
                ("metadata",),
            )
        if self.policy.scan_source_metadata:
            yield from self._iter_metadata_fields(
                payload.get("source", {}).get("metadata", {}),
                ("source", "metadata"),
            )

    def _iter_metadata_fields(
        self,
        value: Any,
        path: tuple[Any, ...],
    ) -> Iterable[tuple[tuple[Any, ...], str]]:
        for nested_path, text in _iter_nested_strings(value, path):
            if (
                self.policy.scan_operational_metadata
                or not _is_operational_metadata_path(nested_path)
            ):
                yield nested_path, text

    @staticmethod
    def _updated_security(
        existing: SecurityMetadata,
        findings: Sequence[SecurityFinding],
    ) -> SecurityMetadata:
        categories = list(existing.sensitive_categories)
        for finding in findings:
            if finding.action is SecurityAction.ALLOW:
                continue
            if finding.category not in categories:
                categories.append(finding.category)

        return existing.model_copy(
            update={
                "contains_sensitive_data": existing.contains_sensitive_data
                or any(f.action is not SecurityAction.ALLOW for f in findings),
                "sensitive_categories": categories,
            },
            deep=True,
        )


def scan_document_security(
    document: NormalizedDocument,
    *,
    scanner: SecurityScanner | None = None,
) -> SecurityScanResult:
    """Convenience wrapper for one-document security scanning."""

    return (scanner or SecurityScanner()).scan(document)


def _highest_action(actions: Iterable[SecurityAction]) -> SecurityAction:
    result = SecurityAction.ALLOW
    for action in actions:
        if _ACTION_PRIORITY[action] > _ACTION_PRIORITY[result]:
            result = action
    return result


def _replacement_for(item: DetectedItem) -> str:
    if item.kind is SecurityFindingKind.PII:
        entity = _slug(item.entity_type or "pii").upper()
        return f"[PII:{entity}]"
    if item.category == "internal_ip":
        return "[INTERNAL_IP]"
    if item.category == "local_path":
        return "[LOCAL_PATH]"
    return "[REDACTED]"


def _apply_redactions(text: str, spans: Sequence[tuple[int, int, str]]) -> str:
    """Apply non-overlapping replacements while tolerating overlapping detectors."""

    valid = sorted(
        (
            (max(0, start), min(len(text), end), replacement)
            for start, end, replacement in spans
            if end > start
        ),
        key=lambda item: (item[0], -(item[1] - item[0])),
    )
    if not valid:
        return text

    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in valid:
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _iter_nested_strings(
    value: Any,
    path: tuple[Any, ...],
) -> Iterable[tuple[tuple[Any, ...], str]]:
    if isinstance(value, str):
        if value:
            yield path, value
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _iter_nested_strings(nested, (*path, key))
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _iter_nested_strings(nested, (*path, index))


def _is_entropy_secret_type(secret_type: str) -> bool:
    return secret_type.strip().lower() in _ENTROPY_SECRET_TYPES


def _has_credential_context(line: str) -> bool:
    return bool(_CREDENTIAL_CONTEXT_RE.search(line))


def _is_operational_metadata_path(path: Sequence[Any]) -> bool:
    # Only metadata/source.metadata paths reach this helper.  Treat the nearest
    # named leaf as the semantic field name so list indices do not matter.
    for part in reversed(path):
        if isinstance(part, str):
            return part.lower() in _OPERATIONAL_METADATA_KEYS
    return False


def _deduplicate_detected_items(items: Sequence[DetectedItem]) -> list[DetectedItem]:
    seen: set[tuple[Any, ...]] = set()
    output: list[DetectedItem] = []
    for item in items:
        key = (
            item.kind,
            item.category,
            item.detector,
            item.field_path,
            item.entity_type,
            item.line_number,
            item.start,
            item.end,
            item.evidence_hash,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _deduplicate_findings(findings: Sequence[SecurityFinding]) -> list[SecurityFinding]:
    seen: set[tuple[Any, ...]] = set()
    output: list[SecurityFinding] = []
    for finding in findings:
        key = (
            finding.kind,
            finding.category,
            finding.detector,
            finding.action,
            finding.field_path,
            finding.entity_type,
            finding.line_number,
            finding.start,
            finding.end,
            finding.evidence_hash,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return output


def _get_path(root: Any, path: Sequence[Any]) -> Any:
    current = root
    for part in path:
        current = current[part]
    return current


def _set_path(root: Any, path: Sequence[Any], value: Any) -> None:
    current = root
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _format_path(path: Sequence[Any]) -> str:
    result = ""
    for part in path:
        if isinstance(part, int):
            result += f"[{part}]"
        elif not result:
            result = str(part)
        else:
            result += f".{part}"
    return result


def _security_summary(
    decision: SecurityAction,
    findings: Sequence[SecurityFinding],
) -> dict[str, Any]:
    category_counts = Counter(finding.category for finding in findings)
    detector_counts = Counter(finding.detector for finding in findings)
    return {
        "schema_version": 1,
        "decision": decision.value,
        "finding_count": len(findings),
        "category_counts": dict(sorted(category_counts.items())),
        "detector_counts": dict(sorted(detector_counts.items())),
        "redacted": any(f.action is SecurityAction.REDACT for f in findings),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ContextMetadataDetector",
    "DetectSecretsDetector",
    "DetectedItem",
    "PresidioPiiDetector",
    "SecurityAction",
    "SecurityDependencyError",
    "SecurityDetector",
    "SecurityFinding",
    "SecurityFindingKind",
    "SecurityPolicy",
    "SecurityScanResult",
    "SecurityScanner",
    "scan_document_security",
]
