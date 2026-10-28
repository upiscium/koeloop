"""Opt-in fixed-fixture CPU STT comparison; no new recordings or remote requests.

Run in the optional parakeet shell. Requires the public synthetic WAVs generated
by previous PoC checks. Each timed request loads a fresh model, like the UI API.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
import unicodedata
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from examples.segmentation_samples import SAMPLES
from examples.stt_live import to_16k
from restream.japanese_stt import PROFILES
from restream.parakeet import EXPORT, MODEL, REVISION
from restream.stt import BACKENDS, create_stt, validate_recording

CASES = [
    ("sentence_1.2x", "artifacts/pauses-1.2-verified/sentence-original.wav", "今日は良い天気です。散歩に出かけます。"),
    ("comma_1.2x", "artifacts/pauses-1.2-verified/comma_inside-original.wav", "今日は良い天気なので、散歩に出かけます。"),
    ("relative", "artifacts/syntax-audio/relative-sentence/combined.wav", dict(SAMPLES)["relative"]),
    ("multi_clause", "artifacts/syntax-audio-multi-clause/multi_clause-sentence/combined.wav", dict(SAMPLES)["multi_clause"]),
    ("llm_response", "artifacts/live-20261021/qwen3.5-4b-live-balanced/combined.wav",
     "窓を開けずに外を眺め、静かな雨音を聴きながら心を落ち着かせてください。\n暖かい飲み物を持って室内でゆっくり読んだり寝たりして気持ちよくなるまで休んでくださいね。"),
]


def normalize(text):
    # Intentionally no conversion of Japanese kanji readings or numbers to words.
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and unicodedata.category(c)[0] not in {"P", "S"})


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


async def trial(args):
    os.environ["STT_BACKEND"] = args.trial
    recording = validate_recording(to_16k(Path(args.audio).read_bytes()))
    service = create_stt()
    if not service.status()["available"]:
        raise RuntimeError(service.status()["message"])
    started = time.monotonic()
    text = await service.transcribe(recording)
    elapsed = (time.monotonic() - started) * 1000
    return {"backend": args.trial, "text": text, "audio_seconds": recording.seconds,
            "pcm_wav_sha256": hashlib.sha256(recording.wav).hexdigest(),
            "request_ms": round(elapsed, 1),
            "rtf": round(elapsed / 1000 / recording.seconds, 4),
            "child_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}


async def compare(args):
    cases = list(CASES)
    if args.additional_manifest:
        manifest = json.loads(Path(args.additional_manifest).read_text(encoding="utf-8"))
        cases.extend((case["name"], case["audio"], case["reference"]) for case in manifest["cases"])
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
        if set(args.cases) != {case[0] for case in cases}:
            raise ValueError("unknown case name")
    if any(not Path(path).is_file() for _, path, _ in cases):
        raise RuntimeError("Generate the documented public WAV fixtures first; see CASES in examples/compare_stt.py")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("onnx-asr", "onnxruntime", "transformers", "torch", "vosk"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    profiles = {"parakeet": {"model": MODEL, "export": EXPORT, "revision": REVISION, "precision": "int8", "decoder": "TDT greedy"},
                "whisper": {"model": "Whisper base Q5_1", "engine": "whisper.cpp", "decoder": "beam 5"}}
    profiles.update({name: {"model": profile.model, "revision": profile.revision,
                            "engine": profile.engine, "precision": "FP32" if name != "vosk" else "model default",
                            "decoder": "beam 5" if name == "whisper-ja" else "greedy CTC (no external LM)" if name == "wav2vec-ja" else "Kaldi graph"}
                     for name, profile in PROFILES.items()})
    report = {"started_utc": datetime.now(UTC).isoformat(), "platform": platform.platform(),
              "python": platform.python_version(), "logical_cpus": os.cpu_count(), "threads": 4,
              "device": "CPU", "dependencies": versions, "backends": args.backends,
              "profiles": {name: profiles[name] for name in args.backends},
              "condition": "One fresh subprocess/model per timed request. Fixture read/resampling excluded. No concurrent test jobs.",
              "sampling": "One warm-up request per backend, then 3 sequential trials per case/backend in alternating order.",
              "normalization": "NFKC + casefold; remove whitespace/punctuation/symbols; no numeral or reading normalization.",
              "limitations": "Small synthetic VOICEVOX:ずんだもん fixture set, not representative microphone speech. Not upstream CER; not GTX1080Ti measurements.",
              "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                  "flake.lock", "restream/stt.py", "restream/japanese_stt_worker.py", "restream/parakeet_worker.py",
                  "nix/japanese-stt-models.nix", "examples/compare_stt.py")},
              "warmup": [], "cases": []}

    async def run(backend, path):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "examples.compare_stt", "--trial", backend, "--audio", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "STT_THREADS": "4"},
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 90)
            if process.returncode:
                raise RuntimeError(stderr.decode("utf-8", errors="replace"))
            return json.loads(stdout)
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()

    for backend in args.backends:
        report["warmup"].append(await run(backend, cases[0][1]))
    for name, path, reference in cases:
        case = {"name": name, "audio": path, "reference": reference, "trials": []}
        for repetition in range(3):
            order = args.backends if repetition % 2 == 0 else args.backends[::-1]
            for backend in order:
                result = await run(backend, path)
                expected, actual = normalize(reference), normalize(result["text"])
                result.update(repetition=repetition + 1, normalized_edits=edit_distance(expected, actual),
                              normalized_reference_chars=len(expected))
                case["trials"].append(result)
        assert len({trial["pcm_wav_sha256"] for trial in case["trials"]}) == 1
        case["summary"] = {}
        for backend in args.backends:
            trials = [trial for trial in case["trials"] if trial["backend"] == backend]
            timings = [trial["request_ms"] for trial in trials]
            summary = {"texts": list(dict.fromkeys(trial["text"] for trial in trials)),
                       "median_ms": statistics.median(timings), "min_ms": min(timings), "max_ms": max(timings),
                       "peak_rss_mib": max(trial["child_peak_rss_mib"] for trial in trials),
                       "median_rtf": statistics.median(trial["rtf"] for trial in trials),
                       "edits": [trial["normalized_edits"] for trial in trials]}
            case["summary"][backend] = summary
        report["cases"].append(case)
        (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"case": name, "summary": case["summary"]}, ensure_ascii=False), flush=True)
    report["finished_utc"] = datetime.now(UTC).isoformat()
    report["status"] = "passed"
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=list(BACKENDS))
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS), default=["whisper", "parakeet"])
    parser.add_argument("--additional-manifest")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--audio")
    parser.add_argument("--output", default="artifacts/stt-comparison-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.trial:
        if not args.audio:
            parser.error("--trial requires --audio")
        print(json.dumps(asyncio.run(trial(args)), ensure_ascii=False))
    else:
        asyncio.run(compare(args))
