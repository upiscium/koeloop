"""Resident Vosk real-time replay diagnostic. No network, mic, LLM, or TTS calls.

One Model is shared across trials. Each utterance gets a new KaldiRecognizer.
Arrival deadlines are based on original audio timestamps, NEVER decoder speed.
"""

import argparse
import hashlib
import io
import json
import math
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
import wave
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from examples.compare_stt import CASES, edit_distance, normalize
from examples.stt_live import to_16k

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000


def percentile(values, percent):
    """Nearest-rank percentile; no inference of confidence intervals."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * percent / 100) - 1)]


def replay(recognizer, pcm, *, chunk_ms=100, tail_ms=2000, clock=time.perf_counter, sleep=time.sleep):
    if len(pcm) % 2 or not pcm or chunk_ms not in {50, 100, 200} or not 0 <= tail_ms <= 5000:
        raise ValueError("invalid PCM or replay settings")
    original_frames = len(pcm) // 2
    original_ms = original_frames / RATE * 1000
    stream = pcm + b"\x00\x00" * (RATE * tail_ms // 1000)
    chunk_bytes = RATE * chunk_ms // 1000 * 2
    started = clock()
    cpu_started = time.process_time()
    rows, events, final_texts = [], [], []
    previous_partial = ""
    for offset in range(0, len(stream), chunk_bytes):
        chunk = stream[offset:offset + chunk_bytes]
        end_seconds = (offset + len(chunk)) / 2 / RATE
        deadline = started + end_seconds  # Chunk cannot arrive before its last sample.
        sleep(max(0, deadline - clock()))
        before = clock()
        lag_ms = max(0, before - deadline) * 1000
        if lag_ms > 2000:
            raise RuntimeError(f"Decoder fell more than 2 seconds behind the audio ({lag_ms:.1f} ms); replay aborted")
        endpoint = bool(recognizer.AcceptWaveform(chunk))
        result = json.loads(recognizer.Result() if endpoint else recognizer.PartialResult())
        after = clock()
        text = result.get("text" if endpoint else "partial", "")
        if not isinstance(text, str):
            raise TypeError("invalid Vosk result")
        rows.append({"audio_end_ms": round(end_seconds * 1000, 3),
                     "service_ms": round((after - before) * 1000, 3),
                     "queue_wait_ms": round(lag_ms, 3),
                     "completion_lag_ms": round(max(0, after - deadline) * 1000, 3)})
        if endpoint:
            if text:
                final_texts.append(text)
                events.append({"type": "final", "text": text, "audio_end_ms": round(end_seconds * 1000, 3),
                               "wall_ms": round((after - started) * 1000, 3)})
            previous_partial = ""
        elif text != previous_partial:
            events.append({"type": "partial", "text": text, "audio_end_ms": round(end_seconds * 1000, 3),
                           "wall_ms": round((after - started) * 1000, 3)})
            previous_partial = text
    before_flush = clock()
    remaining = json.loads(recognizer.FinalResult()).get("text", "")
    ended = clock()
    if remaining:
        final_texts.append(remaining)
        events.append({"type": "forced_final", "text": remaining, "wall_ms": round((ended - started) * 1000, 3)})
    partials = [event for event in events if event["type"] == "partial" and event["text"]]
    after_end = [event for event in events if event["type"] == "final" and event["audio_end_ms"] >= original_ms]
    services = [row["service_ms"] for row in rows]
    cpu_seconds = time.process_time() - cpu_started
    final_text = " ".join(final_texts).strip()
    return {
        "source_ms": round(original_ms, 3), "tail_ms": tail_ms, "chunk_ms": chunk_ms,
        "source_pcm_sha256": hashlib.sha256(pcm).hexdigest(), "frames_consumed": len(stream) // 2,
        "first_partial_ms": partials[0]["wall_ms"] if partials else None,
        "auto_final_after_file_end_ms": round(after_end[0]["wall_ms"] - original_ms, 3) if after_end else None,
        "forced_final_needed": bool(remaining), "flush_service_ms": round((ended - before_flush) * 1000, 3),
        "partial_changes": len(partials), "auto_final_count": sum(event["type"] == "final" for event in events),
        "service_p50_ms": percentile(services, 50), "service_p95_ms": percentile(services, 95),
        "service_max_ms": max(services), "queue_wait_max_ms": max(row["queue_wait_ms"] for row in rows),
        "completion_lag_p95_ms": percentile([row["completion_lag_ms"] for row in rows], 95),
        "completion_lag_max_ms": max(row["completion_lag_ms"] for row in rows),
        "late_over_one_chunk": sum(row["queue_wait_ms"] > chunk_ms for row in rows),
        "processing_rtf": round((sum(services) + (ended - before_flush) * 1000) / (original_ms + tail_ms), 4),
        "cpu_percent_one_core": round(cpu_seconds / max(ended - started, 0.001) * 100, 1),
        "elapsed_ms": round((ended - started) * 1000, 3), "text": final_text,
        "rows": rows, "events": events,
    }


def pcm_from_wav(path):
    with wave.open(io.BytesIO(to_16k(path.read_bytes())), "rb") as source:
        return source.readframes(source.getnframes())


def model_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(b"\x00")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def summary(trials):
    result = {"trial_count": len(trials)}
    for key in ("first_partial_ms", "auto_final_after_file_end_ms", "service_p95_ms", "service_max_ms",
                "queue_wait_max_ms", "completion_lag_p95_ms", "completion_lag_max_ms", "processing_rtf", "cpu_percent_one_core"):
        values = [trial[key] for trial in trials if trial[key] is not None]
        result[key] = {"median": round(statistics.median(values), 4), "min": min(values), "max": max(values)} if values else None
    result["forced_final_trials"] = sum(trial["forced_final_needed"] for trial in trials)
    result["texts"] = list(dict.fromkeys(trial["text"] for trial in trials))
    return result


def run_trial(model, pcm, chunk_ms, competing_audio=None):
    from vosk import KaldiRecognizer

    started = time.perf_counter()
    recognizer = KaldiRecognizer(model, RATE)
    init_ms = (time.perf_counter() - started) * 1000
    competitor = None
    try:
        if competing_audio:
            competitor = subprocess.Popen(
                [sys.executable, "-m", "examples.compare_stt", "--trial", "parakeet", "--audio", competing_audio],
                cwd=ROOT, env={**os.environ, "STT_THREADS": "4", "STT_DEVICE": "cpu", "HF_HUB_OFFLINE": "1"},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        result = replay(recognizer, pcm, chunk_ms=chunk_ms)
        result["recognizer_init_ms"] = round(init_ms, 3)
        if competitor:
            stdout, stderr = competitor.communicate(timeout=90)
            if competitor.returncode:
                raise RuntimeError(f"Competing Parakeet job failed: {stderr.decode(errors='replace')[-2000:]}")
            result["competing_job"] = json.loads(stdout)
        return result
    finally:
        if competitor and competitor.poll() is None:
            # Only our newly created process group (including its recognizer child).
            try:
                os.killpg(competitor.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                competitor.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(competitor.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                competitor.wait(timeout=3)


def run(args):
    os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    directory = Path(os.environ.get("STT_VOSK_MODEL_DIR", ""))
    if not (directory / "am/final.mdl").is_file():
        raise RuntimeError("Use nix develop .#vosk or set STT_VOSK_MODEL_DIR to an installed model")
    if args.competing_parakeet:
        from restream.stt import create_stt

        if not create_stt("parakeet").status()["available"]:
            raise RuntimeError("Competing-load test needs the optional Parakeet environment")
    output = ROOT / args.output
    if not output.resolve().is_relative_to(ROOT / "artifacts") or not output.parent.is_dir():
        raise ValueError("output must be below the existing project artifacts directory")
    if output.exists():
        raise FileExistsError(output)

    from vosk import KaldiRecognizer, Model, SetLogLevel

    cases = [case for case in CASES if case[0] in {"sentence_1.2x", "multi_clause"}]
    manifest = json.loads((ROOT / "artifacts/stt-number-fixtures/manifest.json").read_text())
    cases.extend((case["name"], case["audio"], case["reference"]) for case in manifest["cases"] if case["name"] == "numbers_date")
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
    fixtures = [(name, pcm_from_wav(ROOT / path), reference) for name, path, reference in cases]
    report = {"started_utc": datetime.now(UTC).isoformat(), "host_label": args.host_label,
              "python": platform.python_version(), "platform": platform.platform(), "vosk": version("vosk"),
              "uid": os.getuid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "cpu": subprocess.run(["lscpu"], capture_output=True, text=True, check=True, timeout=10).stdout,
              "load_start": os.getloadavg(), "model": "vosk-model-small-ja-0.22",
              "model_content_sha256": model_digest(directory),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol": {"chunk_ms": args.chunk_ms, "tail_silence_ms": 2000, "repetitions": args.repeats,
                           "resident_model": True, "new_recognizer_per_trial": True, "partial_words": False,
                           "clock": "audio-end timestamp deadlines; late decoder never slows the input clock",
                           "warmup": "one unpaced utterance, excluded", "model_cache": "OS cache not cleared",
                           "fresh_measurement_process": True,
                           "competing_job": "one CPU Parakeet recognition per trial, four threads, fresh model" if args.competing_parakeet else None,
                           "endpoint_delay_reference": "end of original WAV, NOT manually annotated speech end",
                           "first_partial_reference": "time from stream start, NOT per-word ground-truth latency",
                           "processing_rtf_denominator": "source audio plus appended 2s silence",
                           "network_browser_microphone": False, "llm_calls": 0}, "cases": []}
    try:
        SetLogLevel(-1)
        started = time.perf_counter()
        model = Model(model_path=str(directory))
        report["model_load_ms"] = round((time.perf_counter() - started) * 1000, 3)
        warmup = KaldiRecognizer(model, RATE)
        warmup.AcceptWaveform(fixtures[0][1])
        warmup.FinalResult()
        del warmup
        for name, pcm, reference in fixtures:
            trials = []
            case = {"name": name, "reference": reference, "trials": trials}
            report["cases"].append(case)
            for repetition in range(args.repeats):
                audio_path = next(path for case_name, path, _ in cases if case_name == name)
                result = run_trial(model, pcm, args.chunk_ms, audio_path if args.competing_parakeet else None)
                result.update(repetition=repetition + 1,
                              normalized_edits=edit_distance(normalize(reference), normalize(result["text"])))
                trials.append(result)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            case["summary"] = summary(trials)
            print(json.dumps({"case": name, "summary": case["summary"]}, ensure_ascii=False), flush=True)
        if args.sustained_seconds:
            pieces, references = [], []
            frames = 0
            index = 0
            while frames < RATE * args.sustained_seconds:
                _, pcm, reference = fixtures[index % len(fixtures)]
                pieces.append(pcm)
                references.append(reference)
                frames += len(pcm) // 2
                if frames < RATE * args.sustained_seconds:
                    pieces.append(b"\x00\x00" * 12800)  # 0.8s between complete utterances.
                    frames += 12800
                index += 1
            recognizer = KaldiRecognizer(model, RATE)
            result = replay(recognizer, b"".join(pieces), chunk_ms=args.chunk_ms)
            del recognizer
            result["normalized_edits"] = edit_distance(normalize("".join(references)), normalize(result["text"]))
            report["sustained"] = result
            print(json.dumps({"sustained_seconds": result["source_ms"] / 1000, "summary": summary([result])}, ensure_ascii=False), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        report["load_end"] = os.getloadavg()
        report["finished_utc"] = datetime.now(UTC).isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-ms", type=int, choices=[50, 100, 200], default=100)
    parser.add_argument("--repeats", type=int, choices=[1, 3], default=3)
    parser.add_argument("--sustained-seconds", type=int, choices=[0, 60], default=60)
    parser.add_argument("--cases", nargs="+", choices=["sentence_1.2x", "multi_clause", "numbers_date"])
    parser.add_argument("--competing-parakeet", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.competing_parakeet and args.sustained_seconds:
        parser.error("Use --sustained-seconds 0 for the bounded competing-job test")
    if args.worker:
        run(args)
    else:
        # Nix's exec can inherit its launcher's historical ru_maxrss (~1.7GiB).
        # A fresh child ensures reported peak RSS belongs to this experiment.
        result = subprocess.run([sys.executable, "-m", "examples.vosk_stream_check", *sys.argv[1:], "--worker"],
                                cwd=ROOT, timeout=600, check=False)
        raise SystemExit(result.returncode)
