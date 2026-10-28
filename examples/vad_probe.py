"""Opt-in real Silero endpoint diagnostics with public WAVs, no ASR/LLM calls."""

import argparse
import asyncio
import hashlib
import io
import json
import time
import wave
from pathlib import Path

from examples.stt_live import to_16k
from restream.vad import ResidentSilero, VADStart


def pcm(path):
    with wave.open(io.BytesIO(to_16k(Path(path).read_bytes())), "rb") as source:
        return source.readframes(source.getnframes())


def write_report(output, report):
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)


async def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    speech = await asyncio.to_thread(pcm, args.audio)
    cases = [
        ("silence", b"\x00\x00" * 48000, False),
        ("speech", b"\x00\x00" * 8000 + speech + b"\x00\x00" * 32000, True),
        ("hesitation_600ms", speech + b"\x00\x00" * 9600 + speech + b"\x00\x00" * 32000, True),
    ]
    service = ResidentSilero()
    report = {"engine": service.status(), "audio": args.audio, "cases": [], "asr_llm_requests": 0}
    try:
        for name, audio, expected in cases:
            settings = VADStart(type="start")
            turn = await service.start(settings)
            started = time.perf_counter()
            states = []
            for offset in range(0, len(audio), 3200):
                event = await asyncio.to_thread(turn.feed, audio[offset:offset + 3200])
                assert "text" not in event
                if event["type"] == "end":
                    break
                if not states or states[-1] != event["state"]:
                    states.append(event["state"])
            else:
                event = await asyncio.to_thread(turn.end)
            assert event["speech"] is expected
            if name == "hesitation_600ms":
                # Endpoint must be past the second utterance's start, not at the pause.
                assert event["end_sample"] > len(speech) // 2 + 9600
            report["cases"].append({"name": name, "input_frames": len(audio) // 2,
                                    "input_sha256": hashlib.sha256(audio).hexdigest(), "states": states,
                                    "endpoint": event, "replay_compute_ms": round((time.perf_counter() - started) * 1000, 1)})
        report["status"] = "passed"
    finally:
        await service.close()
    await asyncio.to_thread(write_report, output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default="artifacts/stt-number-fixtures/walk.wav")
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
