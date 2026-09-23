from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import pytest
from pydantic import ValidationError

from kb_agent.ingestion.security import (
    ContextMetadataDetector,
    DetectedItem,
    SecurityAction,
    SecurityFindingKind,
    SecurityPolicy,
    SecurityScanner,
)
from kb_agent.models import NormalizedDocument, SecurityMetadata, SourceReference


@dataclass
class FakeSecretDetector:
    token: str = "sk-test-secret"

    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        if self.token not in text:
            return []
        return [
            DetectedItem(
                kind=SecurityFindingKind.SECRET,
                category="secret:fake_api_key",
                detector="fake-secret",
                field_path=field_path,
                entity_type="Fake API Key",
                evidence_hash=sha256(self.token.encode()).hexdigest(),
            )
        ]


@dataclass
class FakePiiDetector:
    token: str = "alice@example.com"

    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        findings: list[DetectedItem] = []
        start = 0
        while True:
            index = text.find(self.token, start)
            if index < 0:
                return findings
            findings.append(
                DetectedItem(
                    kind=SecurityFindingKind.PII,
                    category="pii:email_address",
                    detector="fake-presidio",
                    field_path=field_path,
                    entity_type="EMAIL_ADDRESS",
                    confidence=0.99,
                    start=index,
                    end=index + len(self.token),
                    evidence_hash=sha256(self.token.encode()).hexdigest(),
                )
            )
            start = index + len(self.token)


class EmptyDetector:
    def detect(self, text: str, *, field_path: str) -> list[DetectedItem]:
        return []


def make_document(content: str, **kwargs) -> NormalizedDocument:
    return NormalizedDocument(
        document_id="doc-1",
        title=kwargs.pop("title", "Security test"),
        content=content,
        author=kwargs.pop("author", None),
        source=SourceReference(
            source_type="conversation",
            provider="chatgpt",
            uri="s3://kb-raw/chatgpt/test.json",
            metadata=kwargs.pop("source_metadata", {}),
        ),
        metadata=kwargs.pop("metadata", {}),
        security=kwargs.pop("security", SecurityMetadata()),
        **kwargs,
    )


def scanner(*, policy: SecurityPolicy | None = None) -> SecurityScanner:
    return SecurityScanner(
        policy=policy,
        secret_detector=FakeSecretDetector(),
        pii_detector=FakePiiDetector(),
        context_detector=ContextMetadataDetector(),
    )


def test_secret_blocks_document_without_copying_secret_into_finding() -> None:
    result = scanner().scan(make_document("token = sk-test-secret"))

    assert result.blocked is True
    assert result.document is None
    assert result.decision is SecurityAction.BLOCK
    assert result.findings[0].category == "secret:fake_api_key"
    assert "sk-test-secret" not in result.model_dump_json()
    assert result.security.contains_sensitive_data is True
    assert "secret:fake_api_key" in result.security.sensitive_categories


def test_pii_is_marked_by_default_without_modifying_content() -> None:
    doc = make_document("Contact alice@example.com")
    result = scanner().scan(doc)

    assert result.decision is SecurityAction.MARK
    assert result.document is not None
    assert result.document.content == doc.content
    assert "pii:email_address" in result.document.security.sensitive_categories
    assert result.document.metadata["security_scan"]["decision"] == "mark"


def test_pii_redaction_updates_content_hash() -> None:
    doc = make_document(
        "Contact alice@example.com",
        content_hash=sha256(b"Contact alice@example.com").hexdigest(),
    )
    result = scanner(policy=SecurityPolicy(pii_action=SecurityAction.REDACT)).scan(doc)

    assert result.decision is SecurityAction.REDACT
    assert result.document is not None
    assert result.document.content == "Contact [PII:EMAIL_ADDRESS]"
    assert result.document.content_hash == sha256(result.document.content.encode()).hexdigest()
    assert result.document.content_hash != doc.content_hash


def test_redaction_cleans_duplicate_chat_turn_text_in_metadata() -> None:
    doc = make_document(
        "User email: alice@example.com",
        metadata={
            "turns": [
                {
                    "role": "user",
                    "text": "alice@example.com",
                }
            ]
        },
    )
    result = scanner(policy=SecurityPolicy(pii_action="redact")).scan(doc)

    assert result.document is not None
    assert result.document.metadata["turns"][0]["text"] == "[PII:EMAIL_ADDRESS]"
    assert "alice@example.com" not in result.document.model_dump_json()


def test_redaction_cleans_source_metadata_strings() -> None:
    doc = make_document(
        "safe content",
        source_metadata={"uploaded_by": "alice@example.com"},
    )
    result = scanner(policy=SecurityPolicy(pii_action="redact")).scan(doc)

    assert result.document is not None
    assert result.document.source.metadata["uploaded_by"] == "[PII:EMAIL_ADDRESS]"


def test_private_ip_and_local_paths_are_marked_not_removed() -> None:
    empty = EmptyDetector()
    security_scanner = SecurityScanner(
        secret_detector=empty,
        pii_detector=empty,
        context_detector=ContextMetadataDetector(),
    )
    doc = make_document(
        "Server 192.168.121.78 uses /dev/nvme0n1p4 and /home/user/project/file.txt"
    )
    result = security_scanner.scan(doc)

    assert result.decision is SecurityAction.MARK
    assert result.document is not None
    assert "192.168.121.78" in result.document.content
    assert "internal_ip" in result.security.sensitive_categories
    assert "local_path" in result.security.sensitive_categories


def test_public_ipv4_is_not_classified_as_internal_ip() -> None:
    empty = EmptyDetector()
    security_scanner = SecurityScanner(
        secret_detector=empty,
        pii_detector=empty,
        context_detector=ContextMetadataDetector(),
    )
    result = security_scanner.scan(make_document("DNS 8.8.8.8"))

    assert result.decision is SecurityAction.ALLOW
    assert result.findings == []


def test_context_can_be_redacted_by_policy() -> None:
    empty = EmptyDetector()
    security_scanner = SecurityScanner(
        policy=SecurityPolicy(context_action="redact"),
        secret_detector=empty,
        pii_detector=empty,
        context_detector=ContextMetadataDetector(),
    )
    result = security_scanner.scan(make_document("Host 10.0.0.7 path /etc/hosts"))

    assert result.document is not None
    assert "10.0.0.7" not in result.document.content
    assert "/etc/hosts" not in result.document.content
    assert "[INTERNAL_IP]" in result.document.content
    assert "[LOCAL_PATH]" in result.document.content


def test_existing_acl_and_sensitive_categories_are_preserved() -> None:
    doc = make_document(
        "alice@example.com",
        security=SecurityMetadata(
            tenant_id="company",
            allowed_groups=["aissd"],
            sensitive_categories=["existing"],
        ),
    )
    result = scanner().scan(doc)

    assert result.security.tenant_id == "company"
    assert result.security.allowed_groups == ["aissd"]
    assert result.security.sensitive_categories == ["existing", "pii:email_address"]


def test_secret_detection_inside_metadata_also_blocks() -> None:
    doc = make_document(
        "safe content",
        metadata={"turns": [{"text": "token = sk-test-secret"}]},
    )
    result = scanner().scan(doc)

    assert result.blocked is True
    assert any(f.field_path == "metadata.turns[0].text" for f in result.findings)


def test_secret_action_cannot_be_redact() -> None:
    with pytest.raises(ValidationError):
        SecurityPolicy(secret_action="redact")


def test_scan_flags_can_disable_optional_detectors() -> None:
    # No detect-secrets/Presidio package is needed when those scans are disabled.
    result = SecurityScanner(
        enable_secret_scan=False,
        enable_pii_scan=False,
        enable_context_scan=False,
    ).scan(make_document("alice@example.com token=sk-test-secret"))

    assert result.decision is SecurityAction.ALLOW
    assert result.document is not None
    assert result.findings == []
