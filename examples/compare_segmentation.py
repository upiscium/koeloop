"""Replay fixed texts through three policies; optionally synthesize listening WAVs.

No LLM calls. This diagnostic reports boundaries, not proven perceptual quality.
"""

import argparse
import asyncio
import json
import time
import wave
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx

from examples.live_test import decode_wav
from examples.segmentation_samples import SAMPLES
from restream.providers import Settings, synthesize
from restream.segmenter import Segmenter
from restream.syntax import ANALYZER, MODEL


def replay(text, mode, chunk=4):
    segmenter = Segmenter(mode=mode)
    segments = []
    decisions = []
    first_after = None
    started = time.monotonic()
    for offset in range(0, len(text), chunk):
        found = segmenter.feed(text[offset:offset + chunk])
        if found and first_after is None:
            first_after = min(offset + chunk, len(text))
        segments.extend(found)
        decisions.extend(segmenter.diagnostics)
        segmenter.diagnostics.clear()
    segments.extend(segmenter.flush())
    decisions.extend(segmenter.diagnostics)
    assert "".join(s.text for s in segments) == text
    return {"mode": mode, "segments": [asdict(s) for s in segments],
            "first_segment_after_chars": first_after or len(text),
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "segmentation": segmenter.metrics(), "decisions": decisions}


async def main(args):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ANALYZER.parse("解析器の準備をします。")
    report = {"model": MODEL, "spacy": version("spacy"), "model_version": version("ja-core-news-sm"),
              "model_load_ms": round((time.monotonic() - started) * 1000, 2),
              "chunk_characters": args.chunk, "min_chars": 24, "max_chars": 160, "lookahead": 12,
              "note": "Single diagnostic replay per case; no human listening or quality verdict.", "cases": []}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        if args.audio:
            speakers = await client.get(args.voicevox.rstrip("/") + "/speakers")
            speakers.raise_for_status()
            speaker = next(s for s in speakers.json() if any(style["id"] == 3 for style in s["styles"]))
            (directory / "CREDIT.txt").write_text(f"VOICEVOX:{speaker['name']}\n", encoding="utf-8")
        for name, text in SAMPLES:
            if args.cases and name not in args.cases:
                continue
            result = {"name": name, "text": text, "runs": []}
            for mode in ("sentence", "balanced", "syntax"):
                run = await asyncio.to_thread(replay, text, mode, args.chunk)
                if args.audio:
                    audio_dir = directory / f"{name}-{mode}"
                    audio_dir.mkdir()
                    frames = []
                    for index, segment in enumerate(run["segments"], start=1):
                        if segment["text"].strip():
                            audio = await synthesize(client, Settings(voicevox_url=args.voicevox), segment["text"], 3, 1.0)
                            pcm, _ = decode_wav(audio)
                            frames.append(pcm)
                            (audio_dir / f"{index:03d}.wav").write_bytes(audio)
                    with wave.open(str(audio_dir / "combined.wav"), "wb") as output:
                        output.setnchannels(1)
                        output.setsampwidth(2)
                        output.setframerate(24000)
                        for pcm in frames:
                            output.writeframes(pcm)
                result["runs"].append(run)
                print(json.dumps({"case": name, "mode": mode, "segments": run["segments"],
                                  "metrics": run["segmentation"]}, ensure_ascii=False), flush=True)
            report["cases"].append(result)
            (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artifacts: {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/syntax-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--chunk", type=int, default=4, choices=range(1, 65))
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--voicevox", default="http://127.0.0.1:50021")
    parser.add_argument("--cases", nargs="+", choices=[name for name, _ in SAMPLES])
    asyncio.run(main(parser.parse_args()))
