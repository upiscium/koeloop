"""One local CPU Parakeet trial; no downloads, no LLM calls, no physical mic.

Explicitly supplied audio and transcript are saved only to --output.
Use the optional flake shell: nix develop .#parakeet -c python -m examples.parakeet_probe ...
"""

import argparse
import json
import os
import resource
import time
from importlib.metadata import version
from pathlib import Path

from examples.stt_live import to_16k
from restream.parakeet import EXPORT, MODEL, REVISION
from restream.parakeet_worker import load_model, recognize
from restream.stt import validate_recording


def main(args):
    recording = validate_recording(to_16k(Path(args.audio).read_bytes()))
    model_dir = Path(os.environ["STT_PARAKEET_MODEL_DIR"])
    if not model_dir.is_dir():
        raise ValueError("STT_PARAKEET_MODEL_DIR must be an existing local directory")
    started = time.monotonic()
    model = load_model(model_dir, 4)
    loaded = time.monotonic()
    text = recognize(model, recording)
    finished = time.monotonic()
    report = {"status": "passed" if text else "empty", "audio": args.audio,
              "original_model": MODEL, "export": EXPORT, "revision": REVISION,
              "decoder": "TDT", "quantization": "int8", "provider": "CPUExecutionProvider",
              "threads": 4, "onnx_asr": version("onnx-asr"), "onnxruntime": version("onnxruntime"),
              "audio_seconds": recording.seconds, "text": text,
              "load_ms": round((loaded - started) * 1000, 1),
              "inference_ms": round((finished - loaded) * 1000, 1),
              "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
              "llm_requests": 0, "physical_microphone": False}
    with Path(args.output).open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
