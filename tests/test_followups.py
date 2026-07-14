"""Tests for grounded follow-up suggestion parsing + stream stripping."""

import json

from app.agent.orchestrator import (
    _FollowupStreamGate,
    _parse_followups,
    _split_followups,
)


def _run_gate(deltas):
    """Feed deltas through the gate and return the concatenated visible text."""
    gate = _FollowupStreamGate()
    out = "".join(gate.feed(d) for d in deltas)
    return out + gate.flush()


def test_gate_passes_plain_text_unchanged():
    assert _run_gate(["Xin ", "chào ", "bạn nhé"]) == "Xin chào bạn nhé"


def test_gate_strips_block_in_one_chunk():
    text = 'Câu trả lời đầy đủ.[[FOLLOWUPS]][{"label":"a","query":"b"}]'
    assert _run_gate([text]) == "Câu trả lời đầy đủ."


def test_gate_strips_block_split_across_chunks():
    deltas = ["Trả lời xong rồi.", "[[FOLL", "OWUPS]]", '[{"label":"a","query":"b"}]']
    assert _run_gate(deltas) == "Trả lời xong rồi."


def test_gate_flushes_partial_marker_prefix():
    # Ends on a prefix of the marker that never completes → must survive intact.
    assert _run_gate(["Giá rất tốt ", "[[FOL"]) == "Giá rất tốt [[FOL"


def test_gate_swallows_everything_after_marker():
    gate = _FollowupStreamGate()
    assert gate.feed("Nội dung.[[FOLLOWUPS]][") == "Nội dung."
    assert gate.feed('{"label":"a","query":"b"}]') == ""
    assert gate.flush() == ""


def test_split_followups_extracts_visible_and_chips():
    text = 'Nội dung.[[FOLLOWUPS]][{"label":"Xem thêm","query":"cho tôi xem thêm"}]'
    visible, chips = _split_followups(text)
    assert visible == "Nội dung."
    assert chips == [{"label": "Xem thêm", "query": "cho tôi xem thêm"}]


def test_split_followups_absent_returns_full_text():
    visible, chips = _split_followups("Chỉ có nội dung.")
    assert visible == "Chỉ có nội dung."
    assert chips == []


def test_parse_followups_skips_incomplete_items():
    assert _parse_followups('[{"label":"chỉ label"}, {"query":"chỉ query"}]') == []


def test_parse_followups_rejects_non_json():
    assert _parse_followups("không phải json gì cả") == []


def test_parse_followups_caps_at_four():
    arr = json.dumps([{"label": f"l{i}", "query": f"q{i}"} for i in range(9)])
    assert len(_parse_followups(arr)) == 4


def test_parse_followups_handles_code_fence():
    raw = '```json\n[{"label":"a","query":"b"}]\n```'
    assert _parse_followups(raw) == [{"label": "a", "query": "b"}]
