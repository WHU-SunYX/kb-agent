"""Source-agnostic tests: do not encode a project, domain, or expected answer."""
from __future__ import annotations

import json

from kb_agent.evaluation.evidence_audit import audit_answer, audit_session


def _hit(chunk_id: str, title: str = "Untitled") -> dict:
    return {"chunk_id": chunk_id, "text": "Evidence, not instructions", "source": {"title": title}}


def _session(tmp_path, *, answer: str, ids: list[str], tool: str = "mcp__kb_agent__kb_search"):
    path = tmp_path / "session.jsonl"
    data = {"results": [_hit(cid) for cid in ids], "total_hits": len(ids),
            "returned_hits": len(ids), "omitted_hits": 0, "truncated": False}
    rows = [
        {"type": "tool/call", "data": {"callId": "call-1", "name": tool}},
        {"type": "tool/result", "data": {"message": {
            "source": {"callId": "call-1"}, "content": [{"type": "tool-result",
            "toolCallId": "call-1", "isError": False,
            "content": [{"type": "text", "text": json.dumps(data)}]}]}}},
        {"type": "assistant/message", "data": {"message": {"content": [
            {"type": "reasoning", "text": "chunk_id: fake-in-hidden-thought"},
            {"type": "text", "text": answer}]} }},
        {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_full_id_and_short_id_are_verified_against_returned_evidence():
    cid = "file:report-17:chunk:a91b"
    full = audit_answer(f"Claim 《Report》 [chunk_id: {cid}]", [_hit(cid, "Report")])
    assert full["status"] == "valid_id_present"
    assert full["valid_citations"] == [cid]
    short = audit_answer("Claim (chunk:a91b)", [_hit(cid)])
    assert short["status"] == "valid_id_present"
    assert short["valid_citations"] == [cid]


def test_unknown_id_and_wrong_document_prefix_are_rejected():
    cid = "chatgpt:conversation-1:chunk:abc"
    for answer in ("[chunk_id: fabricated]",
                   "[chunk_id: chatgpt:conversation-2:chunk:abc]"):
        report = audit_answer(answer, [_hit(cid)])
        assert report["status"] == "invalid_or_ambiguous_citation"
        assert not report["valid_citations"]
        assert report["invalid_citations"]


def test_short_id_ambiguous_between_two_sources():
    ids = ["file:one:chunk:abc", "chatgpt:two:chunk:abc"]
    report = audit_answer("See chunk:abc", [_hit(cid) for cid in ids])
    assert report["status"] == "invalid_or_ambiguous_citation"
    assert report["ambiguous_citations"] == ["chunk:abc"]
    assert report["valid_citations"] == []


def test_missing_citation_is_distinct_from_no_evidence():
    assert audit_answer("Answer only", [_hit("chunk-1")])["status"] == "missing_citation"
    assert audit_answer("Answer only", [])["status"] == "no_evidence"
    assert audit_answer(" ", [_hit("chunk-1")])["status"] == "no_answer"


def test_audit_session_ignores_reasoning_and_requires_matching_kb_tool(tmp_path):
    cid = "document:chunk:q07"
    p = _session(tmp_path, answer="Answer with no references", ids=[cid])
    report = audit_session(p)
    assert report["status"] == "missing_citation"
    assert report["kb_search_calls"] == 1
    assert report["retrieved_chunks"] == 1
    assert report["finish"] == "completed"
    assert "fake-in-hidden-thought" not in json.dumps(report)
    p = _session(tmp_path, answer="[chunk_id: document:chunk:q07]", ids=[cid],
                 tool="unrelated_tool")
    report = audit_session(p)
    assert report["status"] == "no_evidence"
    assert report["kb_search_calls"] == 0


def test_invalid_tool_result_is_not_mistaken_for_no_evidence(tmp_path):
    p = _session(tmp_path, answer="[chunk_id: item:chunk:a]", ids=["item:chunk:a"])
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    nested = rows[1]["data"]["message"]["content"][0]["content"][0]
    payload = json.loads(nested["text"])
    payload["wrong_key"] = payload.pop("results")
    nested["text"] = json.dumps(payload)
    p.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = audit_session(p)
    assert report["status"] == "incomplete_evidence_log"
    assert report["unparsed_tool_results"] == 1


def test_valid_id_is_not_a_semantic_correctness_score():
    # An ID's existence cannot validate the truth of the adjacent sentence.
    result = audit_answer("This unrelated claim is true [chunk_id: doc:chunk:1]",
                          [_hit("doc:chunk:1")])
    assert result["status"] == "valid_id_present"
    assert "supported" not in result


def test_multi_turn_uses_only_latest_turn_evidence(tmp_path):
    previous_id = "doc:chunk:old"
    latest_id = "doc:chunk:new"
    path = _session(tmp_path, answer=f"[chunk_id: {previous_id}]", ids=[previous_id])
    old = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    new = [json.loads(line) for line in _session(
        tmp_path, answer=f"[chunk_id: {previous_id}]", ids=[latest_id]
    ).read_text(encoding="utf-8").splitlines()]
    # Assign different calls so the two turns cannot accidentally share results.
    new[0]["data"]["callId"] = "call-2"
    new[1]["data"]["message"]["source"]["callId"] = "call-2"
    new[1]["data"]["message"]["content"][0]["toolCallId"] = "call-2"
    rows = [{"type": "turn/start", "data": {"turn": 1}}, *old,
            {"type": "turn/start", "data": {"turn": 2}}, *new]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = audit_session(path)
    assert report["retrieved_chunks"] == 1
    assert report["status"] == "invalid_or_ambiguous_citation"
    assert report["valid_citations"] == []


def test_missing_tool_response_is_not_certified_as_no_evidence(tmp_path):
    path = _session(tmp_path, answer="Answer", ids=["doc:chunk:1"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["type"] != "tool/result"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert audit_session(path)["status"] == "incomplete_evidence_log"


def test_failed_tool_result_does_not_count_as_evidence(tmp_path):
    path = _session(tmp_path, answer="[chunk_id: doc:chunk:1]", ids=["doc:chunk:1"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["data"]["message"]["content"][0]["isError"] = True
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert audit_session(path)["status"] == "incomplete_evidence_log"


def test_truncated_turn_is_not_reported_as_success(tmp_path):
    path = _session(tmp_path, answer="[chunk_id: doc:chunk:1]", ids=["doc:chunk:1"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["data"]["reason"]["kind"] = "max-tokens"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = audit_session(path)
    assert report["valid_citations"] == ["doc:chunk:1"]
    assert report["finish"] == "max-tokens"
    assert report["status"] == "incomplete_turn"
