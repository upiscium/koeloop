"""Local Japanese STT worker. No remote code, text rewriting, or digit suppression."""

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

from .japanese_stt import PROFILES
from .stt import MAX_BYTES, MAX_TEXT, validate_recording


def recognize_vosk(directory, recording):
    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(-1)
    # Explicit model_path avoids Vosk's model-name download mechanism.
    model = Model(model_path=str(directory))
    recognizer = KaldiRecognizer(model, 16000)
    texts = []
    with wave.open(io.BytesIO(recording.wav), "rb") as source:
        while pcm := source.readframes(4000):
            if recognizer.AcceptWaveform(pcm):
                texts.append(json.loads(recognizer.Result())["text"])
    texts.append(json.loads(recognizer.FinalResult())["text"])
    # Keep decoded words/numbers as-is, including Japanese inter-word spaces.
    return " ".join(text for text in texts if text).strip()


def recognize_transformers(backend, directory, recording, threads, device="cpu", gpu_memory_fraction=0.16, metrics=None):
    import numpy as np
    import torch
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    prepare_device(torch, device, gpu_memory_fraction)
    started = time.monotonic()
    with wave.open(io.BytesIO(recording.wav), "rb") as source:
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    model_args = {"local_files_only": True, "trust_remote_code": False, "weights_only": True,
                  "use_safetensors": False, "dtype": torch.float32, "attn_implementation": "eager"}
    # The official fixed revisions publish .bin, not safetensors. Enforce restricted
    # weights-only loading with current PyTorch rather than enabling arbitrary pickle.
    if backend == "whisper-ja":
        processor = WhisperProcessor.from_pretrained(directory, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(directory, **model_args).eval()
        model.generation_config.forced_decoder_ids = None
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        if device == "cuda":
            model = model.to(device)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            torch.cuda.synchronize()
        loaded = time.monotonic()
        with torch.inference_mode():
            ids = model.generate(**inputs, language="ja", task="transcribe", return_timestamps=False,
                                 do_sample=False, num_beams=5, max_new_tokens=400)
        # Whisper's generation wrapper strips EOS from returned segments. Do not
        # require EOS in this tensor. Conservatively reject near-cap output (the
        # wrapper may also remove several prompt/special tokens).
        validate_whisper_length(ids.shape[-1])
        record_metrics(torch, device, metrics, started, loaded)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    processor = Wav2Vec2Processor.from_pretrained(directory, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(directory, **model_args).eval()
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    if device == "cuda":
        model = model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        torch.cuda.synchronize()
    loaded = time.monotonic()
    with torch.inference_mode():
        ids = model(**inputs).logits.argmax(dim=-1)
    record_metrics(torch, device, metrics, started, loaded)
    return processor.batch_decode(ids)[0].strip()


def prepare_device(torch, device, fraction):
    if device not in {"cpu", "cuda"} or not 0 < fraction <= 0.5:
        raise ValueError("unsupported device or unsafe memory fraction")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
        torch.cuda.set_per_process_memory_fraction(fraction)
        torch.cuda.reset_peak_memory_stats()


def record_metrics(torch, device, metrics, started, loaded):
    if device == "cuda":
        torch.cuda.synchronize()
    if metrics is not None:
        metrics.update(device=device, load_and_prepare_ms=round((loaded - started) * 1000, 1),
                       inference_ms=round((time.monotonic() - loaded) * 1000, 1))
        if device == "cuda":
            metrics.update(cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                           cuda_peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1))


def validate_whisper_length(length):
    if length >= 390:
        raise ValueError("Whisper output is near its token cap; use a shorter recording")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=list(PROFILES), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.16)
    args = parser.parse_args()
    if not all((args.model_dir / filename).is_file() for filename in PROFILES[args.backend].files):
        raise ValueError("Local model files are missing")
    recording = validate_recording(sys.stdin.buffer.read(MAX_BYTES + 1))
    if recording.rms < 0.001:
        return
    text = (recognize_vosk(args.model_dir, recording) if args.backend == "vosk" else
            recognize_transformers(args.backend, args.model_dir, recording, args.threads,
                                   args.device, args.gpu_memory_fraction))
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError("Invalid recognition result")
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - do not leak audio, transcripts, or server paths to stderr.
        sys.stderr.write("Japanese STT failed. Check the configured local model and runtime.\n")
        raise SystemExit(1) from None
