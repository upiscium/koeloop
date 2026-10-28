"""Optional Parakeet JA ONNX backend; no ML imports in the API server process."""

import os
import sys
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

from .stt import MAX_SECONDS, SAMPLE_RATE, ProcessSTT, STTConfig

MODEL = "nvidia/parakeet-tdt_ctc-0.6b-ja"
EXPORT = "OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx"
REVISION = "4353e7b9e2e9ebdd35e85b8140c7351d03c2219c"
MODEL_FILES = ("encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "config.json", "vocab.txt")


class ParakeetSTT(ProcessSTT):
    def __init__(self, config=None):
        self.config = config if config is not None else replace(
            STTConfig.from_env(), command=sys.executable,
            model_path=os.getenv("STT_PARAKEET_MODEL_DIR", ""))

    def status(self):
        available = bool(self.config.model_path and all(
            Path(self.config.model_path, name).is_file() for name in MODEL_FILES
        ) and find_spec("onnx_asr") and find_spec("onnxruntime"))
        return {"available": available, "backend": "parakeet", "engine": "onnx-asr",
                "engine_label": "Parakeet JA 0.6B / ONNX int8・CPU", "model": MODEL,
                "language": "ja", "max_seconds": MAX_SECONDS, "sample_rate": SAMPLE_RATE,
                "message": "Parakeet 日本語モデルでローカル認識します。自動送信しません。" if available else
                "Parakeet が未設定です。nix develop .#parakeet でサーバーを起動し直してください。"}

    def command_args(self):
        return (self.config.command, "-m", "restream.parakeet_worker",
                "--model-dir", self.config.model_path, "--threads", str(self.config.threads))

    def process_environment(self):
        return {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "OMP_NUM_THREADS": str(self.config.threads), "OPENBLAS_NUM_THREADS": str(self.config.threads)}
