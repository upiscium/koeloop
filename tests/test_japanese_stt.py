import asyncio
import io
import json
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from examples.compare_stt import edit_distance, normalize
from restream import japanese_stt
from restream.app import create_app
from restream.japanese_stt import PROFILES, JapaneseSTT
from restream.japanese_stt_worker import (
    prepare_device,
    recognize_transformers,
    recognize_vosk,
    validate_whisper_length,
)
from restream.stt import BACKENDS, STTConfig, create_stt, validate_recording


def wav():
    data = io.BytesIO()
    with wave.open(data, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x20\x20" * 16000)
    return data.getvalue()


@pytest.mark.parametrize("backend", list(PROFILES))
def test_optional_profile_is_local_and_isolated(monkeypatch, backend):
    profile = PROFILES[backend]
    monkeypatch.setenv(profile.environment, "/private/model")
    service = create_stt(backend)
    assert isinstance(service, JapaneseSTT)
    assert service.config.model_path == "/private/model"
    assert backend in service.command_args()
    assert "private" not in str(service.status())
    env = service.process_environment()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
    monkeypatch.setattr(japanese_stt, "find_spec", lambda _: object())
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert service.status()["available"] is True
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert service.status()["available"] is False


def test_vosk_is_path_only_and_preserves_actual_numbers(monkeypatch):
    paths, pcm_inputs = [], []

    class Recognizer:
        def __init__(self, model, rate):
            assert model == "local-model" and rate == 16000

        def AcceptWaveform(self, data):
            pcm_inputs.append(data)
            return len(pcm_inputs) == 2

        def Result(self):
            return json.dumps({"text": "ペン 3 本"})

        def FinalResult(self):
            return json.dumps({"text": "30 分 後 に 散歩"})

    def model(*, model_path):
        paths.append(model_path)
        return "local-model"

    monkeypatch.setitem(sys.modules, "vosk", SimpleNamespace(Model=model, KaldiRecognizer=Recognizer, SetLogLevel=lambda _: None))
    result = recognize_vosk(Path("/local/vosk"), validate_recording(wav()))
    assert paths == ["/local/vosk"]
    assert b"".join(pcm_inputs) == b"\x20\x20" * 16000
    assert result == "ペン 3 本 30 分 後 に 散歩"


def test_whisper_wrapper_can_return_short_sequence_without_eos():
    validate_whisper_length(20)
    with pytest.raises(ValueError, match="cap"):
        validate_whisper_length(390)


@pytest.mark.parametrize("backend", ["whisper-ja", "wav2vec-ja"])
def test_transformers_loading_and_decoding_contract(monkeypatch, backend):
    calls = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Tensor:
        shape = (1, 20)

        def argmax(self, dim):
            assert dim == -1
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True}
            calls.append(("processor", path))
            return cls()

        def __call__(self, samples, **kwargs):
            assert kwargs["sampling_rate"] == 16000
            assert kwargs["return_attention_mask"] is True
            return {"input_features" if backend == "whisper-ja" else "input_values": Tensor()}

        def batch_decode(self, ids, **kwargs):
            return ["ペンを3本買って、30分後に散歩します。"]

    class Model:
        generation_config = SimpleNamespace(forced_decoder_ids=[[1, 123]], suppress_tokens=[1, 2])

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["weights_only"] is True
            assert kwargs["trust_remote_code"] is False
            assert kwargs["local_files_only"] is True
            assert kwargs["dtype"] == "float32"
            calls.append(("model", path))
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["language"] == "ja" and kwargs["task"] == "transcribe"
            assert kwargs["num_beams"] == 5 and kwargs["do_sample"] is False
            assert "suppress_tokens" not in kwargs
            assert self.generation_config.suppress_tokens == [1, 2]
            assert self.generation_config.forced_decoder_ids is None
            return Tensor()

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=Tensor())

    # NumPy is installed with the syntax extra in flake; tests also run without it.
    pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None,
        float32="float32", inference_mode=Context))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        WhisperProcessor=Processor, Wav2Vec2Processor=Processor,
        WhisperForConditionalGeneration=Model, Wav2Vec2ForCTC=Model))
    assert recognize_transformers(backend, Path("/local/model"), validate_recording(wav()), 4) == "ペンを3本買って、30分後に散歩します。"
    assert len(calls) == 2


def test_backend_header_selects_only_server_registered_services():
    class Fake:
        def __init__(self, name, available=True):
            self.name, self.available, self.calls = name, available, 0

        def status(self):
            return {"backend": self.name, "available": self.available, "message": "未導入"}

        async def transcribe(self, recording):
            self.calls += 1
            return self.name + " 3本"

    default, other, absent = Fake("whisper"), Fake("vosk"), Fake("wav2vec-ja", False)
    with TestClient(create_app(stt=default)) as client:
        client.app.state.stt_backends.update(vosk=other, **{"wav2vec-ja": absent})
        headers = {"Content-Type": "audio/wav", "X-STT-Backend": "vosk"}
        response = client.post("/api/stt/transcribe", content=wav(), headers=headers)
        assert response.status_code == 200
        assert response.json()["text"] == "vosk 3本"
        assert default.calls == 0 and other.calls == 1
        headers["X-STT-Backend"] = "https://not-a-backend.example"
        assert client.post("/api/stt/transcribe", content=wav(), headers=headers).status_code == 400
        headers["X-STT-Backend"] = "wav2vec-ja"
        assert client.post("/api/stt/transcribe", content=wav(), headers=headers).status_code == 503
        assert absent.calls == default.calls == 0
        assert not client.app.state.stt_busy
        assert {entry["id"] for entry in client.get("/api/config").json()["stt_backends"]} == {"whisper", "vosk", "wav2vec-ja"}


def test_normalization_does_not_remove_numbers():
    assert normalize("ペンを３本、３０分。") == "ペンを3本30分"
    assert edit_distance(normalize("3本"), normalize("散歩")) > 0
    assert set(PROFILES).issubset(BACKENDS)


def test_silence_in_new_profiles_does_not_spawn(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("must not start a process for silence")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    silent = replace(validate_recording(wav()), rms=0)
    for backend in PROFILES:
        service = JapaneseSTT(backend, STTConfig(model_path="/fake"))
        monkeypatch.setattr(service, "status", lambda: {"available": True})
        assert asyncio.run(service.transcribe(silent)) == ""


def test_cuda_is_opt_in_and_memory_bounded(monkeypatch):
    monkeypatch.setenv("STT_DEVICE", "cuda")
    monkeypatch.setenv("STT_GPU_MEMORY_FRACTION", "0.16")
    service = JapaneseSTT("whisper-ja", STTConfig(model_path="/fake"))
    assert service.device == "cuda"
    assert "CUDA FP32" in service.status()["engine_label"]
    assert service.command_args()[-4:] == ("--device", "cuda", "--gpu-memory-fraction", "0.16")
    assert JapaneseSTT("vosk", STTConfig(model_path="/fake")).device == "cpu"


@pytest.mark.parametrize(("device", "fraction"), [("wrong", 0.16), ("cuda", 0), ("cuda", 0.9), ("cuda", float("nan"))])
def test_invalid_cuda_settings_rejected_before_torch_call(device, fraction):
    with pytest.raises(ValueError):
        prepare_device(SimpleNamespace(), device, fraction)


def test_requested_cuda_never_silently_falls_back_to_cpu():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="no silent CPU fallback"):
        prepare_device(torch, "cuda", 0.16)


def test_cuda_memory_fraction_applied_before_loading():
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        set_per_process_memory_fraction=lambda value: calls.append(value),
        reset_peak_memory_stats=lambda: calls.append("reset")))
    prepare_device(torch, "cuda", 0.16)
    assert calls == [0.16, "reset"]
