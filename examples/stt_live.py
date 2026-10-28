"""Opt-in local STT check using an existing WAV; never starts a mic or sends to an LLM."""

import argparse
import asyncio
import io
import json
import os
import sys
import time
import wave
from array import array
from pathlib import Path

import httpx

from examples.live_test import running_server
from restream.providers import Settings
from restream.stt import STTConfig, create_stt, validate_recording


def to_16k(data):
    with wave.open(io.BytesIO(data), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError("test input must be mono 16-bit PCM")
        rate = source.getframerate()
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    # Test fixture conversion only. Browser capture uses Web Audio's resampler.
    result = array("h")
    for index in range(round(len(samples) * 16000 / rate)):
        position = index * rate / 16000
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        result.append(round(samples[left] + (samples[right] - samples[left]) * (position - left)))
    if sys.byteorder != "little":
        result.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(result.tobytes())
    return output.getvalue()


async def main(args):
    recording = validate_recording(to_16k(Path(args.audio).read_bytes()))
    if args.diagnose:
        if os.getenv("STT_BACKEND", "whisper") != "whisper":
            raise ValueError("--diagnose is whisper-only; use examples.parakeet_probe for Parakeet diagnostics")
        config = STTConfig.from_env()
        process = await asyncio.create_subprocess_exec(
            config.command, "--model", config.model_path, "--file", "-", "--language", "ja",
            "--threads", "4", "--no-gpu", "--no-timestamps",
            "--output-txt", "--output-file", "-",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(recording.wav)
        return {"seconds": recording.seconds, "rms": recording.rms, "returncode": process.returncode,
                "stdout": stdout.decode(), "stderr": stderr.decode()}
    service = create_stt()
    started = time.monotonic()
    if args.api:
        async with running_server(Settings()) as base, httpx.AsyncClient(timeout=90, trust_env=False) as client:
            response = await client.post(base + "/api/stt/transcribe", content=recording.wav,
                                         headers={"Content-Type": "audio/wav"})
            response.raise_for_status()
            text = response.json()["text"]
    else:
        text = await service.transcribe(recording)
    if not text:
        raise RuntimeError("no transcript returned")
    result = {"engine": service.status().get("engine"), "backend": service.status().get("backend"),
              "status": "passed", "source": args.audio,
              "seconds": recording.seconds, "transcript": text,
              "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
              "llm_requests": 0, "physical_microphone_tested": False,
              "transport": "HTTP API" if args.api else "native service"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnose", action="store_true", help="record native logs for the explicitly supplied test WAV")
    mode.add_argument("--api", action="store_true", help="start a temporary loopback API server and transcribe through HTTP")
    args = parser.parse_args()
    result = asyncio.run(main(args))
    with Path(args.output).open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
