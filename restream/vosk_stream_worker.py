"""Historical Vosk streaming experiment, not used by the voice-conversation API.

Kept for reproducing earlier diagnostics; current voice mode uses Silero VAD.
"""

import argparse
import base64
import json
import math
import sys
from array import array

from .stt import MAX_TEXT, SAMPLE_RATE


def respond(event):
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    from vosk import KaldiRecognizer, Model, SetLogLevel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    SetLogLevel(-1)
    model = Model(model_path=args.model_dir)
    respond({"type": "loaded"})
    recognizer = None
    heard = False
    while line := sys.stdin.buffer.readline(16385):
        if len(line) > 16384 or not line.endswith(b"\n"):
            raise ValueError("oversized command")
        command = json.loads(line)
        kind = command.get("type")
        if kind == "start":
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
            heard = False
            respond({"type": "ready"})
        elif kind in {"audio", "end"} and recognizer is not None:
            if kind == "audio":
                pcm = base64.b64decode(command["pcm"], validate=True)
                if not 0 < len(pcm) <= 6400 or len(pcm) % 2:
                    raise ValueError("invalid PCM")
                samples = array("h")
                samples.frombytes(pcm)
                if sys.byteorder != "little":
                    samples.byteswap()
                heard |= math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768 > 0.001
                endpoint = bool(recognizer.AcceptWaveform(pcm))
                result = json.loads(recognizer.Result() if endpoint else recognizer.PartialResult())
                text = result.get("text" if endpoint else "partial", "") if heard else ""
                # Blank/silence endpoints don't end a user's turn.
                final = endpoint and bool(text.strip())
            else:
                text = json.loads(recognizer.FinalResult()).get("text", "") if heard else ""
                final = True
            if not isinstance(text, str) or len(text) > MAX_TEXT:
                raise ValueError("invalid transcript")
            respond({"type": "final" if final else "partial", "text": text})
            if final:
                recognizer = None
        else:
            raise ValueError("invalid command state")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - native errors must not expose audio/transcripts/paths.
        sys.stderr.write("Vosk streaming failed. Check local model/runtime.\n")
        raise SystemExit(1) from None
