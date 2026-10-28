"""One local Parakeet recognition per child process, WAV stdin -> UTF-8 stdout."""

import argparse
import io
import sys
import wave
from pathlib import Path

from .parakeet import MODEL_FILES
from .stt import MAX_BYTES, MAX_TEXT, validate_recording


def load_model(directory: Path, threads: int):
    import onnxruntime as ort
    from onnx_asr.loader import Manager

    if not directory.is_dir() or not all((directory / name).is_file() for name in MODEL_FILES):
        raise ValueError("local model files are missing")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    ort.disable_telemetry_events()
    # Existing, explicit local directory + offline=True: no Hub lookup or model code execution.
    return Manager(sess_options=options, providers=["CPUExecutionProvider"]).create_asr(
        "nemo-conformer-tdt", local_dir=directory, quantization="int8", offline=True)


def recognize(model, recording):
    import numpy as np

    with wave.open(io.BytesIO(recording.wav), "rb") as source:
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    text = model.recognize(samples, sample_rate=16000)
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError("invalid recognizer output")
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    recording = validate_recording(sys.stdin.buffer.read(MAX_BYTES + 1))
    if recording.rms < 0.001:
        return
    model = load_model(args.model_dir, args.threads)
    sys.stdout.buffer.write(recognize(model, recording).encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - never print user audio/text or private paths from native errors.
        sys.stderr.write("Parakeet recognition failed. Check local model/runtime configuration.\n")
        raise SystemExit(1) from None
