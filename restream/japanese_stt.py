"""Optional Japanese model profiles. Model paths come only from server configuration."""

import os
import sys
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path

from .stt import MAX_SECONDS, SAMPLE_RATE, ProcessSTT, STTConfig


@dataclass(frozen=True)
class Profile:
    model: str
    revision: str
    environment: str
    engine: str
    label: str
    modules: tuple[str, ...]
    files: tuple[str, ...]


PROFILES = {
    "whisper-ja": Profile(
        "Ivydata/whisper-small-japanese", "d889b2b90cae2e7df49238fb84f7ad13248fc1cd",
        "STT_WHISPER_JA_MODEL_DIR", "transformers", "Whisper-small 日本語微調整 / CPU FP32",
        ("torch", "transformers"),
        ("pytorch_model.bin", "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt")),
    "wav2vec-ja": Profile(
        "Ivydata/wav2vec2-large-xlsr-53-japanese", "3feaaf2fd0e2139ca3343c5a7058e172225579b4",
        "STT_WAV2VEC_JA_MODEL_DIR", "transformers", "wav2vec2 XLSR 日本語微調整 / CPU FP32・CTC",
        ("torch", "transformers"),
        ("pytorch_model.bin", "config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json")),
    "vosk": Profile(
        "vosk-model-small-ja-0.22", "0.22", "STT_VOSK_MODEL_DIR", "vosk",
        "Vosk-small 日本語 0.22 / CPU", ("vosk",),
        ("am/final.mdl", "conf/model.conf", "conf/mfcc.conf", "graph/HCLr.fst", "graph/Gr.fst")),
}


class JapaneseSTT(ProcessSTT):
    def __init__(self, backend, config=None):
        self.backend = backend
        self.profile = PROFILES[backend]
        self.config = config if config is not None else replace(
            STTConfig.from_env(), command=sys.executable, model_path=os.getenv(self.profile.environment, ""))
        self.device = "cpu" if backend == "vosk" else os.getenv("STT_DEVICE", "cpu")
        self.gpu_memory_fraction = os.getenv("STT_GPU_MEMORY_FRACTION", "0.16")

    def status(self):
        device_valid = self.device in {"cpu", "cuda"}
        try:
            device_valid &= 0 < float(self.gpu_memory_fraction) <= 0.5
        except ValueError:
            device_valid = False
        available = bool(device_valid and self.config.model_path and all(
            Path(self.config.model_path, filename).is_file() for filename in self.profile.files
        ) and all(find_spec(module) is not None for module in self.profile.modules))
        return {"available": available, "backend": self.backend, "engine": self.profile.engine,
                "engine_label": self.profile.label.replace("CPU", "CUDA") if self.device == "cuda" else self.profile.label,
                "device": self.device, "model": self.profile.model, "language": "ja",
                "max_seconds": MAX_SECONDS, "sample_rate": SAMPLE_RATE,
                "message": "ローカルで文字起こしします。自動送信しません。" if available else
                f"この認識器は未設定です。モデル・STT_DEVICE を確認するか、nix develop .#{self.backend} で起動してください。"}

    def command_args(self):
        return (self.config.command, "-m", "restream.japanese_stt_worker", "--backend", self.backend,
                "--model-dir", self.config.model_path, "--threads", str(self.config.threads),
                "--device", self.device, "--gpu-memory-fraction", self.gpu_memory_fraction)

    def process_environment(self):
        return {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1", "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": str(self.config.threads), "OPENBLAS_NUM_THREADS": str(self.config.threads)}
