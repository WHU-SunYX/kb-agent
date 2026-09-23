"""Offline, source-agnostic citation checks for Harness session JSONL logs.

Checks only mechanical provenance: whether a final answer cites chunk IDs that
actually appeared in successful kb_search tool responses from the same turn.
It cannot verify whether a passage supports a claim, determine a project's
implementation status, or grade answer correctness. It makes no model/API calls,
and is never imported by the online agent or MCP server.

Usage:
    python -m kb_agent.evaluation.evidence_audit session.jsonl --out audit.json
    python -m kb_agent.evaluation.evidence_audit ./sessions --out audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

KB_TOOL = "mcp__kb_agent__kb_search"

# Citation IDs may be emitted in full, as "chunk:<suffix>", or as a
# uniquely identifying opaque suffix in an explicit citation context.
# Only a full ID satisfies the "exact chunk_id" output contract.
LABELED_ID = re.compile(r"\bchunk_id\s*[:=]\s*[`\"']?([^\s`\"'\]\)）}>;,，。；]+)", re.I)
CHUNK_TOKEN = re.compile(
    r"(?<![\w:./@-])(?:\.\.\.:|(?:[A-Za-z0-9_.@/-]+:)*)chunk:[A-Za-z0-9_.-]+(?![\w:./@-])"
)
# Bare opaque IDs are recognized from the *actual returned IDs*, not from a
# hard-coded hash format. Require citation-like formatting to avoid flagging
# unrelated code identifiers in ordinary text.
BACKTICK_TOKEN = re.compile(r"`([A-Za-z0-9_.-]{3,128})`")
BRACKET_TOKEN = re.compile(r"\[([A-Za-z0-9_.-]{3,128})\]")
ID_BOUNDARY = r"[\w:./@-]"


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_blocks(content: Any) -> list[str]:
    """Handle structured Harness blocks and plain text, omitting reasoning."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
                and isinstance(block.get("text"), str)]
    return []


def _tool_payloads(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Parse MCP text responses without retaining or exporting passage bodies."""
    result: list[dict[str, Any]] = []
    message = _map(_map(record.get("data")).get("message"))
    for outer in message.get("content", []):
        if not isinstance(outer, dict):
            continue
        if outer.get("type") == "tool-result" and outer.get("isError") is True:
            continue
        blocks = outer.get("content", []) if outer.get("type") == "tool-result" else [outer]
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                try:
                    value = json.loads(block["text"])
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict) and isinstance(value.get("results"), list):
                    result.append(value)
    return result


def _tool_call_id(record: Mapping[str, Any]) -> str | None:
    message = _map(_map(record.get("data")).get("message"))
    source = _map(message.get("source"))
    call_id = source.get("callId")
    if isinstance(call_id, str):
        return call_id
    for block in message.get("content", []):
        if isinstance(block, dict) and isinstance(block.get("toolCallId"), str):
            return block["toolCallId"]
    return None


def _mentioned_ids(answer: str, evidence_ids: set[str]) -> set[str]:
    """Find complete IDs on token boundaries, not arbitrary substrings."""
    return {cid for cid in evidence_ids if re.search(
        rf"(?<!{ID_BOUNDARY}){re.escape(cid)}(?!{ID_BOUNDARY})", answer
    )}


def audit_answer(answer: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check mechanical citation provenance against this turn's returned hits.

    Short citations are *traceable* only if they uniquely match a returned ID;
    they are never counted as exact/full-ID compliant. No semantic claim
    verification, project-specific rules, or online model calls are involved.
    """
    evidence: dict[str, dict[str, Any]] = {}
    for hit in hits:
        if isinstance(hit, dict) and isinstance(hit.get("chunk_id"), str):
            cid = hit["chunk_id"].strip()
            if cid:
                evidence[cid] = hit

    base: dict[str, Any] = {
        "retrieved_chunks": len(evidence), "valid_citations": [],
        "full_id_citations": [], "abbreviated_citations": [],
        "invalid_citations": [], "ambiguous_citations": [],
        "citation_format": "none", "full_id_compliant": False,
    }
    if not answer.strip():
        return dict(base, status="no_answer")
    if not evidence:
        return dict(base, status="no_evidence")

    full = _mentioned_ids(answer, set(evidence))
    shortened: dict[str, str] = {}
    invalid: set[str] = set()
    ambiguous: set[str] = set()

    tokens = {m.group(1) for m in LABELED_ID.finditer(answer)}
    tokens.update(m.group(0) for m in CHUNK_TOKEN.finditer(answer))
    known_suffixes = {cid.rsplit(":", 1)[-1] for cid in evidence}
    for match in BACKTICK_TOKEN.finditer(answer):
        candidate = match.group(1)
        if candidate in known_suffixes:
            tokens.add(candidate)
        else:
            # Unknown bare IDs are only mechanically identifiable as citations
            # inside an explicit source bracket (e.g. [《Title》 `opaque-id`]).
            opening = answer.rfind("[", max(0, match.start() - 160), match.start())
            previous_close = answer.rfind("]", max(0, match.start() - 160), match.start())
            closing = answer.find("]", match.end(), match.end() + 160)
            next_open = answer.find("[", match.end(), match.end() + 160)
            if (opening > previous_close and closing >= 0
                    and (next_open < 0 or closing < next_open)):
                context = answer[opening:closing + 1]
                if "《" in context or "chunk_id" in context or "chunk:" in context:
                    tokens.add(candidate)
    for match in BRACKET_TOKEN.finditer(answer):
        if match.group(1) in known_suffixes:
            tokens.add(match.group(1))

    for raw in tokens:
        candidate = raw.replace("\\:", ":").rstrip(".:;")
        if candidate in evidence:
            full.add(candidate)
            continue
        # An ellipsis is *only* shorthand when it exactly prefixes chunk:<ID>;
        # an incorrect document prefix must not be forgiven by suffix matching.
        if candidate.startswith("...:chunk:"):
            short = candidate[len("...:"):]
        elif candidate.startswith("chunk:"):
            short = candidate
        elif ":" not in candidate and re.fullmatch(r"[A-Za-z0-9_.-]{3,128}", candidate):
            short = candidate
        else:
            invalid.add(raw)
            continue
        matches = [cid for cid in evidence
                   if (cid.endswith(":" + short) if short.startswith("chunk:")
                       else cid.rsplit(":", 1)[-1] == short)]
        if len(matches) == 1:
            shortened[raw] = matches[0]
        elif len(matches) > 1:
            ambiguous.add(raw)
        else:
            invalid.add(raw)

    valid = full | set(shortened.values())
    if invalid or ambiguous:
        status = "invalid_or_ambiguous_citation"
    elif valid:
        status = "valid_id_present"  # Backwards-compatible provenance status.
    else:
        status = "missing_citation"
    base.update({
        "status": status,
        "valid_citations": sorted(valid),
        "full_id_citations": sorted(full),
        "abbreviated_citations": [
            {"reference": raw, "chunk_id": cid}
            for raw, cid in sorted(shortened.items())
        ],
        "invalid_citations": sorted(invalid),
        "ambiguous_citations": sorted(ambiguous),
        "citation_format": ("mixed" if full and shortened else
                            "full" if full else "abbreviated" if shortened else "none"),
        "full_id_compliant": bool(full) and not (shortened or invalid or ambiguous),
    })
    return base


def audit_session(path: str | Path) -> dict[str, Any]:
    """Read one Harness log; a session may contain multiple tool calls."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if isinstance(data, dict):
                    records.append(data)

    # An exported session can contain many turns. Judge the last turn only:
    # a citation from a previous turn must not validate this turn's answer.
    starts = [i for i, r in enumerate(records) if r.get("type") == "turn/start"]
    if starts:
        records = records[starts[-1]:]

    call_ids = {str(_map(r.get("data")).get("callId")) for r in records
                if r.get("type") == "tool/call"
                and _map(r.get("data")).get("name") == KB_TOOL
                and isinstance(_map(r.get("data")).get("callId"), str)}
    tool_responses = [r for r in records if r.get("type") == "tool/result"
                      and _tool_call_id(r) in call_ids]
    # Compaction may re-emit tool/result records with truncated JSON. Evaluate
    # each callId once, preferring the first successfully parsed original.
    # A corrupt *duplicate* cannot invalidate an already complete response.
    payload_by_call: dict[str, list[dict[str, Any]]] = {}
    for record in tool_responses:
        call_id = _tool_call_id(record)
        if call_id not in payload_by_call:
            payloads = _tool_payloads(record)
            if payloads:
                payload_by_call[call_id] = payloads
    hits: list[dict[str, Any]] = []
    for payloads in payload_by_call.values():
        for payload in payloads:
            hits.extend(hit for hit in payload["results"] if isinstance(hit, dict))
    # Count *unresolved calls*, not malformed duplicate records.
    unresolved_calls = call_ids - payload_by_call.keys()
    parse_failures = len(unresolved_calls)

    assistants = [r for r in records if r.get("type") == "assistant/message"]
    # Only the latest assistant message's visible text; exclude reasoning and
    # intermediate messages to prevent phantom citations from earlier turns.
    answer = "\n".join(_text_blocks(_map(_map(assistants[-1].get("data")).get("message")).get("content"))) if assistants else ""
    report = audit_answer(answer, hits)
    ends = [r for r in records if r.get("type") == "turn/end"]
    finish = _map(_map(ends[-1].get("data")).get("reason")).get("kind") if ends else None
    report.update({"file": str(path), "kb_search_calls": len(call_ids),
                   "kb_search_results": len(payload_by_call),
                   "tool_result_records": len(tool_responses),
                   "ignored_duplicate_tool_results": max(0, len(tool_responses) - len(payload_by_call)),
                   "unparsed_tool_results": parse_failures, "finish": finish,
                   "answer_present": bool(answer.strip())})
    if parse_failures:
        # Missing/failed responses mean the known evidence set is incomplete.
        report["status"] = "incomplete_evidence_log"
    elif finish != "completed":
        # Even a genuine citation cannot certify a truncated final answer.
        report["status"] = "incomplete_turn"
    return report


def _paths(arguments: list[str]) -> list[Path]:
    result: list[Path] = []
    for value in arguments:
        p = Path(value)
        if p.is_file():
            result.append(p)
        elif p.is_dir():
            result.extend(sorted(p.rglob("*.jsonl")))
        else:
            raise ValueError(f"not a file or directory: {p}")
    return list(dict.fromkeys(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Harness session.jsonl files or directories")
    parser.add_argument("--out", help="Optional local JSON report path")
    args = parser.parse_args(argv)
    try:
        paths = _paths(args.paths)
        if not paths:
            raise ValueError("no session JSONL files found")
        reports = [audit_session(p) for p in paths]
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for report in reports:
        print(f"{report['file']}: {report['status']}; "
              f"traceable={len(report['valid_citations'])} "
              f"full={len(report['full_id_citations'])} "
              f"abbreviated={len(report['abbreviated_citations'])} "
              f"invalid={len(report['invalid_citations'])} "
              f"full_id_compliant={report['full_id_compliant']} "
              f"evidence={report['retrieved_chunks']} finish={report['finish']}")
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
