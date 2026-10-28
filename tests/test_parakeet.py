import asyncio
import io
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from restream import parakeet
from restream.app import create_app
from restream.parakeet import ParakeetSTT
from restream.parakeet_worker import load_model
from restream.stt import STTConfig, STTError, WhisperSTT, create_stt, validate_recording


def recording():
    data = io.BytesIO()
    with wave.open(data, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x20\x20" * 16000)
    return validate_recording(data.getvalue())


def test_default_backend_is_still_whisper(monkeypatch):
    monkeypatch.delenv("STT_BACKEND", raising=False)
    assert isinstance(create_stt(), WhisperSTT)


def test_selected_parakeet_does_not_reuse_whisper_model(monkeypatch):
    monkeypatch.setenv("STT_BACKEND", "parakeet")
    monkeypatch.setenv("STT_MODEL_PATH", "/private/whisper.bin")
    monkeypatch.setenv("STT_PARAKEET_MODEL_DIR", "/private/parakeet")
    service = create_stt()
    assert isinstance(service, ParakeetSTT)
    assert service.config.model_path == "/private/parakeet"
    assert service.command_args() == (sys.executable, "-m", "restream.parakeet_worker",
                                      "--model-dir", "/private/parakeet", "--threads", "4")
    assert service.process_environment()["HF_HUB_OFFLINE"] == "1"
    assert service.process_environment()["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert "private" not in str(service.status())


def test_missing_parakeet_is_reported_not_silently_replaced(monkeypatch):
    monkeypatch.setenv("STT_BACKEND", "parakeet")
    monkeypatch.delenv("STT_PARAKEET_MODEL_DIR", raising=False)
    with TestClient(create_app()) as client:
        status = client.get("/api/config").json()["stt"]
        assert status["available"] is False
        assert status["backend"] == "parakeet"
        response = client.post("/api/stt/transcribe", content=recording().wav, headers={"Content-Type": "audio/wav"})
        assert response.status_code == 503
        assert ".#parakeet" in response.json()["error"]


def test_misspelled_backend_is_unavailable(monkeypatch):
    monkeypatch.setenv("STT_BACKEND", "private-unknown-value")
    status = create_stt().status()
    assert status["available"] is False
    assert "private" not in str(status)


def test_installed_status_checks_all_files_and_dependencies(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: self.name in parakeet.MODEL_FILES)
    monkeypatch.setattr(parakeet, "find_spec", lambda _: object())
    service = ParakeetSTT(STTConfig(command=sys.executable, model_path="/fake/model"))
    assert service.status()["available"] is True
    monkeypatch.setattr(Path, "is_file", lambda self: self.name != "vocab.txt")
    assert service.status()["available"] is False


def test_loader_is_cpu_local_int8_and_offline(monkeypatch):
    calls = []
    options = SimpleNamespace()

    class Manager:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def create_asr(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "local-model"

    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=lambda: options, disable_telemetry_events=lambda: None))
    monkeypatch.setitem(sys.modules, "onnx_asr.loader", SimpleNamespace(Manager=Manager))
    assert load_model(Path("/local/model"), 3) == "local-model"
    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 1
    assert calls[0]["providers"] == ["CPUExecutionProvider"]
    assert calls[1] == (("nemo-conformer-tdt",), {"local_dir": Path("/local/model"), "quantization": "int8", "offline": True})


@pytest.mark.parametrize("cancel", [False, True])
def test_parakeet_timeout_and_cancel_kill_own_process(monkeypatch, cancel):
    async def check():
        entered = asyncio.Event()
        finished = asyncio.Event()

        class Process:
            returncode = None
            killed = False
            waited = False
            stdout = asyncio.StreamReader()

            class stdin:
                @staticmethod
                def write(data):
                    assert data[:4] == b"RIFF"

                @staticmethod
                async def drain():
                    pass

                @staticmethod
                def close():
                    pass

            async def wait(self):
                await finished.wait()
                self.waited = True
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9
                self.stdout.feed_eof()
                finished.set()

        child = Process()

        async def spawn(*args, **kwargs):
            assert args[1:3] == ("-m", "restream.parakeet_worker")
            assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
            entered.set()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        service = ParakeetSTT(STTConfig(command=sys.executable, model_path="/local/model", timeout=0.03 if not cancel else 5))
        monkeypatch.setattr(service, "status", lambda: {"available": True})
        task = asyncio.create_task(service.transcribe(recording()))
        await entered.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(STTError) as error:
                await task
            assert error.value.status == 504
        assert child.killed and child.waited
    asyncio.run(check())
