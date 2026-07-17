"""Tests for the streaming repetition circuit-breaker (_RepetitionGuard).

Gemini flash occasionally degenerates into an endless run of one character
(a markdown divider ``------``) or repeats an identical chunk. LiteLLM only
kills such a stream after ~100 identical chunks, surfacing as a hard
``MidStreamFallbackError`` after a flood of junk already reached the UI. The
guard trips far earlier and withholds the runaway text.
"""

from app.agent.orchestrator import _FOLLOWUPS_INSTRUCTION, _RepetitionGuard


def _run_guard(chunks):
    """Feed chunks through the guard; return (visible_text, tripped)."""
    guard = _RepetitionGuard()
    out = "".join(guard.feed(c) for c in chunks)
    return out + guard.flush(), guard.tripped


def test_plain_text_passes_through_unchanged():
    text, tripped = _run_guard(["Xin ", "chào ", "bạn nhé"])
    assert text == "Xin chào bạn nhé"
    assert tripped is False


def test_short_divider_survives_intact():
    # A legitimate short run (< threshold) followed by text must not be eaten.
    text, tripped = _run_guard(["Kết quả:\n", "---\n", "còn nữa"])
    assert text == "Kết quả:\n---\ncòn nữa"
    assert tripped is False


def test_runaway_dash_run_trips_and_is_dropped():
    # The exact failure mode: a huge dash run in one chunk.
    text, tripped = _run_guard(["Đây là bảng:\n", "-" * 200])
    assert tripped is True
    assert "Đây là bảng:" in text
    # The runaway run must never reach the client.
    assert "-" * 41 not in text


def test_runaway_run_accumulates_across_small_deltas():
    # Same degeneration, but streamed one dash at a time — must still trip
    # before flooding, and emit (almost) none of the dashes.
    guard = _RepetitionGuard()
    emitted = "".join(guard.feed("-") for _ in range(80))
    emitted += guard.flush()
    assert guard.tripped is True
    assert emitted.count("-") < _RepetitionGuard._MAX_RUN


def test_identical_chunk_repetition_trips():
    guard = _RepetitionGuard()
    for _ in range(_RepetitionGuard._MAX_REPEATS + 5):
        guard.feed("cùng một chunk")
    assert guard.tripped is True


def test_whitespace_run_does_not_trip():
    # A run of whitespace is benign — the guard exempts it so genuine
    # blank-line formatting is never swallowed.
    text, tripped = _run_guard(["Xong.", "\n" * 100])
    assert tripped is False
    assert text == "Xong." + "\n" * 100


def test_once_tripped_stays_sealed():
    guard = _RepetitionGuard()
    for _ in range(60):
        guard.feed("=")
    assert guard.tripped is True
    assert guard.feed("nội dung sau khi trip") == ""
    assert guard.flush() == ""


def test_trips_below_litellm_limit():
    # Both thresholds must sit safely under LiteLLM's 100-chunk guard so we
    # stop the stream before it raises InternalServerError.
    assert _RepetitionGuard._MAX_RUN < 100
    assert _RepetitionGuard._MAX_REPEATS < 100


def test_followups_instruction_forbids_repetition():
    # Prompt hardening (A): the followups block must never be preceded by a
    # divider, and character repetition is explicitly banned.
    assert "KHÔNG lặp lại cùng một ký tự" in _FOLLOWUPS_INSTRUCTION
    assert "`---`" in _FOLLOWUPS_INSTRUCTION
