"""Explicit local model diagnostic. Use public WAV fixtures; native logs are visible."""

import argparse
import json
import os
import time
from pathlib import Path

from examples.stt_live import to_16k
from restream.japanese_stt import PROFILES
from restream.japanese_stt_worker import recognize_transformers, recognize_vosk
from restream.stt import validate_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=list(PROFILES))
    parser.add_argument("audio")
    args = parser.parse_args()
    recording = validate_recording(to_16k(Path(args.audio).read_bytes()))
    directory = Path(os.environ[PROFILES[args.backend].environment])
    started = time.monotonic()
    text = (recognize_vosk(directory, recording) if args.backend == "vosk" else
            recognize_transformers(args.backend, directory, recording, 4))
    print(json.dumps({"backend": args.backend, "text": text,
                      "elapsed_ms": round((time.monotonic() - started) * 1000, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
