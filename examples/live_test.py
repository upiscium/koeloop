"""Opt-in real Ollama -> WebSocket -> VOICEVOX test. Never collected by pytest.

Run from the project root: nix develop -c python -m examples.live_test
This contacts the explicitly configured LLM and writes test audio under artifacts/.
Playback acknowledgements are duration-paced simulations, NOT sound-device playback.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import platform
import socket
import time
import wave
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx
import uvicorn
from websockets.asyncio.client import connect

from restream.app import create_app
from restream.providers import Settings, synthesize

PROMPT = (
    "音声合成の接続テストです。雨の日の過ごし方を、日本語の短い二文、"
    "合計60文字から90文字程度で紹介してください。各文は句点で終え、"
    "本文だけを出力してください。見出し、箇条書き、英語は不要です。"
)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decode_wav(data):
    with wave.open(io.BytesIO(data), "rb") as audio:
        assert audio.getnchannels() == 1, "expected mono WAV"
        assert audio.getframerate() == 24000, "expected 24 kHz WAV"
        assert audio.getsampwidth() == 2, "expected 16-bit PCM"
        assert audio.getnframes() > 0, "empty WAV"
        frames = audio.readframes(audio.getnframes())
        return frames, audio.getnframes() / audio.getframerate()


@asynccontextmanager
async def running_server(settings, client=None):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        create_app(settings, client=client), log_level="warning", ws_max_size=65536, lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("server exited before startup")
                await asyncio.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=15)
        finally:
            sock.close()


async def run_session(base, directory, source, mode, speaker, text=None):
    directory.mkdir()
    started = time.monotonic()
    events = []
    frames = []
    audio_ids = []
    segments = []
    deltas = []
    echo = asyncio.Queue()
    first_delta = first_segment = first_audio = None
    playhead = 0
    gaps = 0
    input_task = None
    acks = []
    success = False

    async def acknowledge(ws, seq, end):
        await asyncio.sleep(max(0, end - time.monotonic()))
        await ws.send(json.dumps({"type": "played", "seq": seq}))

    async def send_input(ws, session_id):
        assert text is not None
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            async def send(event):
                if source == "text":
                    await ws.send(json.dumps(event, ensure_ascii=False))
                    if event["type"] == "text":
                        await asyncio.wait_for(echo.get(), timeout=120)
                else:
                    while True:
                        response = await client.post(f"{base}/api/sessions/{session_id}/events", json=event)
                        if response.status_code != 429:
                            response.raise_for_status()
                            assert response.status_code == 202
                            return
                        await asyncio.sleep(1)

            for offset in range(0, len(text), 8):
                await send({"type": "text", "text": text[offset:offset + 8]})
                await asyncio.sleep(0.1)
            await send({"type": "end"})

    try:
        async with asyncio.timeout(240), connect(
            base.replace("http://", "ws://") + "/ws", max_size=24 * 1024 * 1024, proxy=None,
        ) as ws:
            await ws.send(json.dumps({"type": "start", "source": source, "mode": mode,
                                      "speaker": speaker, "audio": True,
                                      "prompt": PROMPT if source == "llm" else ""}))
            async for message in ws:
                event = json.loads(message)
                elapsed = round((time.monotonic() - started) * 1000, 1)
                event["client_ms"] = elapsed
                kind = event["type"]
                if kind == "audio":
                    audio = base64.b64decode(event.pop("wav_base64"), validate=True)
                    pcm, duration = decode_wav(audio)
                    frames.append(pcm)
                    audio_ids.append(event["seq"])
                    filename = f"{event['seq']:04d}.wav"
                    (directory / filename).write_bytes(audio)
                    event.update(file=filename, duration_s=duration, bytes=len(audio),
                                 sha256=hashlib.sha256(audio).hexdigest())
                    if first_audio is None:
                        first_audio = elapsed
                    now = time.monotonic()
                    if playhead:
                        gaps += max(0, now - playhead)
                    playhead = max(now, playhead) + duration
                    acks.append(asyncio.create_task(acknowledge(ws, event["seq"], playhead)))
                events.append(event)
                if kind == "ready" and source != "llm":
                    input_task = asyncio.create_task(send_input(ws, event["session_id"]))
                elif kind == "delta":
                    deltas.append(event["text"])
                    if source == "text":
                        echo.put_nowait(None)
                    if first_delta is None:
                        first_delta = elapsed
                elif kind == "segment":
                    segments.append(event)
                    if first_segment is None:
                        first_segment = elapsed
                elif kind == "error":
                    raise RuntimeError(event["message"])
                elif kind == "done":
                    success = True
                    break
                if input_task and input_task.done():
                    await input_task
            if input_task:
                await input_task
            await asyncio.gather(*acks)
        assert success, "connection ended without done"
        received = "".join(deltas)
        assert received.strip(), "no generated text"
        assert "".join(s["text"] for s in segments) == received, "text loss or duplication"
        if text is not None:
            assert received == text, "replay differed from input"
        assert audio_ids == [s["seq"] for s in segments if s["text"].strip()], "audio order mismatch"
        assert frames and any(any(frame) for frame in frames), "audio is entirely silent"
        with wave.open(str(directory / "combined.wav"), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            for frame in frames:
                audio.writeframes(frame)
        result = {
            "status": "passed", "source": source, "mode": mode, "text": received,
            "characters": len(received), "segments": len(segments),
            "first_delta_ms": first_delta, "first_segment_ms": first_segment,
            "first_audio_ms": first_audio, "audio_duration_s": round(sum(len(f) for f in frames) / 48000, 3),
            "simulated_queue_gaps_s": round(gaps, 3),
            "synthesis_ms": [e["synthesis_ms"] for e in events if e["type"] == "audio"],
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            "forced_segments": sum(s["reason"] == "forced_limit" for s in segments),
        }
        write_json(directory / "result.json", result)
        print(json.dumps({"run": directory.name, **result}, ensure_ascii=False), flush=True)
        return result
    finally:
        for task in [input_task, *acks]:
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in [input_task, *acks] if t), return_exceptions=True)
        write_json(directory / "events.json", events)


async def main(args):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "dependencies": {name: version(name) for name in ["fastapi", "pydantic", "uvicorn", "httpx", "websockets"]},
        "llm_base_url": args.ollama.rstrip("/") + "/v1", "voicevox_url": args.voicevox,
        "reasoning_effort": args.reasoning_effort or "provider default",
        "voicevox_image": os.getenv("VOICEVOX_IMAGE", "unknown"), "prompt": PROMPT,
        "playback": "duration-paced simulated ACK; no browser or sound device",
        "measurement": "single-run integration diagnostics, not a comparative performance benchmark",
        "replay": {"chunk_characters": 8, "interval_ms": 100}, "runs": [],
    }
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            async def get(url):
                response = await client.get(url)
                response.raise_for_status()
                return response.json()

            models = (await get(args.ollama.rstrip("/") + "/api/tags"))["models"]
            metadata["ollama_version"] = await get(args.ollama.rstrip("/") + "/api/version")
            metadata["voicevox_version"] = await get(args.voicevox.rstrip("/") + "/version")
            speakers = await get(args.voicevox.rstrip("/") + "/speakers")
            speaker = next(s for s in speakers if any(style["id"] == args.speaker for style in s["styles"]))
            metadata["speaker"] = {"id": args.speaker, "name": speaker["name"]}
            (directory / "CREDIT.txt").write_text(f"VOICEVOX:{speaker['name']}\n", encoding="utf-8")
            metadata["models"] = []
            for name in args.models:
                model = next(m for m in models if m["name"] == name)
                assert "completion" in model.get("capabilities", []), f"{name} is not a generation model"
                metadata["models"].append({"name": name, "digest": model["digest"],
                                           "quantization": model["details"].get("quantization_level")})
            # Separate Engine model loading from the LLM-path diagnostics.
            warmup = await synthesize(client, Settings(voicevox_url=args.voicevox), "接続テストです。", args.speaker, 1.0)
            decode_wav(warmup)
            (directory / "warmup.wav").write_bytes(warmup)

        for index, name in enumerate(args.models):
            settings = Settings(voicevox_url=args.voicevox, llm_base_url=metadata["llm_base_url"],
                                llm_model=name, llm_reasoning_effort=args.reasoning_effort)
            async with running_server(settings) as base:
                label = name.replace(":", "-").replace("/", "-")
                result = await run_session(base, directory / f"{label}-live-balanced", "llm", "balanced", args.speaker)
                metadata["runs"].append({"model": name, "directory": f"{label}-live-balanced", **result})
                write_json(directory / "report.json", metadata)
                # Fix the generated text before replaying. No extra LLM requests for comparisons.
                if index == 0 and not args.no_replay:
                    for source, mode in [("text", "balanced"), ("external", "sentence"), ("external", "raw")]:
                        run_name = f"{label}-replay-{source}-{mode}"
                        replay = await run_session(base, directory / run_name, source, mode, args.speaker, result["text"])
                        metadata["runs"].append({"model": name, "directory": run_name, **replay})
                        write_json(directory / "report.json", metadata)
        metadata["status"] = "passed"
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(directory / "report.json", metadata)
        print(f"Artifacts: {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--voicevox", default="http://127.0.0.1:50021")
    parser.add_argument("--models", nargs="+", default=["qwen3.5:4b", "gemma4:12b"])
    parser.add_argument("--speaker", type=int, default=3)
    parser.add_argument("--reasoning-effort", default=os.getenv("LLM_REASONING_EFFORT", ""))
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--output", default="artifacts/live-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    asyncio.run(main(parser.parse_args()))
