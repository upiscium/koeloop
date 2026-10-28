"""Compare original and adjusted VOICEVOX pauses at the same speed and text.

Each text's AudioQuery is fetched once, then copied for the two conditions.
Identical queries reuse the original synthesized WAV before appending PCM silence.
No LLM calls, syntax analysis, browser sleeps, or PCM silence trimming are used.
"""

import argparse
import asyncio
import copy
import json
import math
import sys
import wave
from array import array
from datetime import UTC, datetime
from pathlib import Path

import httpx

from examples.live_test import decode_wav
from restream.pauses import PauseSettings, append_sentence_silence, apply_pauses

CASES = {
    "sentence": ["今日は良い天気です。", "散歩に出かけます。"],
    "comma_inside": ["今日は良い天気なので、散歩に出かけます。"],
    "comma_split": ["今日は良い天気なので、", "散歩に出かけます。"],
    "mixed_quote": ["「良い天気です。今日は、散歩に出かけます。」と話しました。"],
}


def nominal(query):
    scale = query["speedScale"]
    override = query.get("pauseLength")
    pause_scale = query.get("pauseLengthScale", 1.0)
    return {
        "pre_ms": query["prePhonemeLength"] / scale * 1000,
        "post_ms": query["postPhonemeLength"] / scale * 1000,
        "internal_pause_ms": [
            (override if override is not None else phrase["pause_mora"]["vowel_length"]) * pause_scale / scale * 1000
            for phrase in query["accent_phrases"] if phrase.get("pause_mora")
        ],
    }


def audio_stats(pcm):
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    windows = []
    for start in range(0, len(samples), 240):  # 10 ms at 24 kHz
        window = samples[start:start + 240]
        rms = math.sqrt(sum(sample * sample for sample in window) / len(window)) / 32768
        windows.append((start, start + len(window), rms))
    # Neural vocoders can leave very low-level noise in nominal silence. Use a
    # relative threshold and report it; this is not a perceptual silence detector.
    threshold = max(0.0001, max(window[2] for window in windows) * 0.01)
    audible = [(start, end) for start, end, rms in windows if rms > threshold]
    assert audible, "waveform is entirely below the silence threshold"
    return {"duration_ms": len(samples) / 24, "rms_threshold": threshold,
            "leading_low_amplitude_ms": audible[0][0] / 24,
            "trailing_low_amplitude_ms": (len(samples) - audible[-1][1]) / 24}


async def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = PauseSettings(args.sentence_extra_ms, args.comma_scale)
    report = {"speed": args.speed, "speaker_id": 3,
              "settings": {"sentence_extra_ms": settings.sentence_extra_ms, "comma_scale": settings.comma_scale},
              "measurement": "Nominal query durations plus approximate 10ms RMS windows at max(0.0001, peak-window-RMS * 0.01); not human listening.",
              "cases": {}}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.get(args.voicevox.rstrip("/") + "/version")
        response.raise_for_status()
        report["engine_version"] = response.json()
        response = await client.get(args.voicevox.rstrip("/") + "/speakers")
        response.raise_for_status()
        speaker = next(s for s in response.json() if any(style["id"] == 3 for style in s["styles"]))
        (output / "CREDIT.txt").write_text(f"VOICEVOX:{speaker['name']}\n", encoding="utf-8")
        for name, texts in CASES.items():
            queries = []
            for text in texts:
                response = await client.post(args.voicevox.rstrip("/") + "/audio_query", params={"text": text, "speaker": 3})
                response.raise_for_status()
                q = response.json()
                q.update(speedScale=args.speed, outputSamplingRate=24000, outputStereo=False)
                queries.append(q)
            result = {"texts": texts, "conditions": {}}
            original_wavs = []
            for condition in ("original", "adjusted"):
                frames = []
                measurements = []
                for index, (text, source_query) in enumerate(zip(texts, queries, strict=True)):
                    q = copy.deepcopy(source_query)
                    info = apply_pauses(q, text, settings) if condition == "adjusted" else None
                    if condition == "adjusted" and q == source_query:
                        original_audio = original_wavs[index]
                    else:
                        response = await client.post(args.voicevox.rstrip("/") + "/synthesis", params={"speaker": 3}, json=q)
                        response.raise_for_status()
                        original_audio = response.content
                    if condition == "original":
                        original_wavs.append(original_audio)
                    audio = (append_sentence_silence(original_audio, text, settings)
                             if condition == "adjusted" else original_audio)
                    pcm, _ = decode_wav(audio)
                    if name == "sentence" and condition == "adjusted":
                        original_pcm, _ = decode_wav(original_wavs[index])
                        assert pcm == original_pcm + b"\x00\x00" * (24 * settings.sentence_extra_ms)
                    frames.append(pcm)
                    (output / f"{name}-{condition}-{index + 1}.wav").write_bytes(audio)
                    (output / f"{name}-{condition}-{index + 1}.query.json").write_text(
                        json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
                    measurements.append({"nominal": nominal(q), "wave": audio_stats(pcm), "adjustment": info})
                with wave.open(str(output / f"{name}-{condition}.wav"), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    for pcm in frames:
                        wav.writeframes(pcm)
                result["conditions"][condition] = {"segments": measurements}
                if len(measurements) == 2:
                    result["conditions"][condition].update(
                        nominal_join_gap_ms=measurements[0]["nominal"]["post_ms"] + measurements[1]["nominal"]["pre_ms"]
                        + (measurements[0]["adjustment"]["sentence_extra_ms"] if measurements[0]["adjustment"] else 0),
                        measured_join_low_amplitude_ms=measurements[0]["wave"]["trailing_low_amplitude_ms"]
                        + measurements[1]["wave"]["leading_low_amplitude_ms"],
                    )
            report["cases"][name] = result
            print(json.dumps({"case": name, **result}, ensure_ascii=False), flush=True)

    sentence = report["cases"]["sentence"]["conditions"]
    delta = sentence["adjusted"]["nominal_join_gap_ms"] - sentence["original"]["nominal_join_gap_ms"]
    assert math.isclose(delta, settings.sentence_extra_ms)
    for old, new in zip(sentence["original"]["segments"], sentence["adjusted"]["segments"], strict=True):
        duration_delta = new["wave"]["duration_ms"] - old["wave"]["duration_ms"]
        assert math.isclose(duration_delta, settings.sentence_extra_ms, abs_tol=1e-6)
    comma = report["cases"]["comma_inside"]["conditions"]
    old_pause = comma["original"]["segments"][0]["nominal"]["internal_pause_ms"]
    new_pause = comma["adjusted"]["segments"][0]["nominal"]["internal_pause_ms"]
    assert len(old_pause) == len(new_pause) == 1
    assert math.isclose(new_pause[0], old_pause[0] * settings.comma_scale)
    report["status"] = "passed"
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voicevox", default="http://127.0.0.1:50021")
    parser.add_argument("--speed", type=float, default=1.2)
    parser.add_argument("--sentence-extra-ms", type=int, default=200)
    parser.add_argument("--comma-scale", type=float, default=0.8)
    parser.add_argument("--output", default="artifacts/pauses-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not 0.5 <= args.speed <= 2:
        parser.error("speed must be between 0.5 and 2")
    asyncio.run(main(args))
