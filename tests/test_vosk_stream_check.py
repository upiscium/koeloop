import json

import pytest

from examples.vosk_stream_check import percentile, replay


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.value += seconds


class Recognizer:
    def __init__(self, clock, service=0.01, endpoint_at=4):
        self.clock, self.service, self.endpoint_at = clock, service, endpoint_at
        self.count = 0
        self.audio = bytearray()

    def AcceptWaveform(self, pcm):
        self.audio.extend(pcm)
        self.count += 1
        self.clock.value += self.service
        return self.count == self.endpoint_at

    def PartialResult(self):
        return json.dumps({"partial": "" if self.count < 2 or self.count >= self.endpoint_at else "ペン 3 本"})

    def Result(self):
        return json.dumps({"text": "ペン 3 本"})

    def FinalResult(self):
        self.clock.value += 0.005
        return json.dumps({"text": "" if self.count >= self.endpoint_at else "ペン 3 本"})


def test_real_time_deadlines_partial_and_endpoint_are_distinct():
    clock = Clock()
    rec = Recognizer(clock)
    pcm = b"\x01\x00" * 4800  # 300ms
    result = replay(rec, pcm, tail_ms=200, clock=clock.now, sleep=clock.sleep)
    assert bytes(rec.audio) == pcm + b"\x00\x00" * 3200
    assert result["first_partial_ms"] == pytest.approx(210)
    assert result["auto_final_after_file_end_ms"] == pytest.approx(110)
    assert result["service_max_ms"] == pytest.approx(10)
    assert result["queue_wait_max_ms"] == pytest.approx(0)
    assert result["flush_service_ms"] == pytest.approx(5)
    assert result["text"] == "ペン 3 本"
    assert not result["forced_final_needed"]
    assert result["elapsed_ms"] == pytest.approx(515)


def test_slow_decoder_cannot_hide_backlog_by_slowing_input():
    clock = Clock()
    rec = Recognizer(clock, service=0.15, endpoint_at=99)
    result = replay(rec, b"\x01\x00" * 4800, tail_ms=0, clock=clock.now, sleep=clock.sleep)
    assert [row["audio_end_ms"] for row in result["rows"]] == [100, 200, 300]
    assert [row["queue_wait_ms"] for row in result["rows"]] == pytest.approx([0, 50, 100])
    assert result["completion_lag_max_ms"] == pytest.approx(250)
    assert result["processing_rtf"] > 1
    assert result["auto_final_after_file_end_ms"] is None
    assert result["forced_final_needed"]


def test_empty_partial_is_not_reported_as_first_text():
    clock = Clock()
    rec = Recognizer(clock, endpoint_at=1)
    result = replay(rec, b"\x01\x00" * 1600, tail_ms=0, clock=clock.now, sleep=clock.sleep)
    assert result["first_partial_ms"] is None
    assert result["text"] == "ペン 3 本"


def test_overload_aborts_instead_of_dropping_chunks():
    clock = Clock()
    rec = Recognizer(clock, service=3)
    with pytest.raises(RuntimeError, match="fell more than"):
        replay(rec, b"\x01\x00" * 4800, tail_ms=0, clock=clock.now, sleep=clock.sleep)
    assert rec.count == 1


def test_partial_final_chunk_preserves_exact_sample_count():
    clock = Clock()
    rec = Recognizer(clock)
    result = replay(rec, b"\x01\x00" * 4321, tail_ms=0, clock=clock.now, sleep=clock.sleep)
    assert result["frames_consumed"] == 4321
    assert result["rows"][-1]["audio_end_ms"] == pytest.approx(270.062, abs=0.001)


def test_nearest_rank_percentile():
    assert percentile([], 95) is None
    assert percentile([2, 1, 3, 100], 95) == 100
    assert percentile([2, 1, 3, 100], 50) == 2
