import asyncio
import io
import threading
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from restream.app import create_app
from restream.stt import MAX_BYTES, STTConfig, STTError, WhisperSTT, validate_recording


def wav(seconds=1, rate=16000, channels=1, width=2, silent=False):
    data = io.BytesIO()
    with wave.open(data, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes((b"\x00" * width if silent else b"\x20" * width) * channels * int(seconds * rate))
    return data.getvalue()


class FakeSTT:
    def __init__(self, text="今日は良い天気です。", available=True):
        self.text, self.available = text, available
        self.calls = []

    def status(self):
        return {"available": self.available, "engine": "test", "max_seconds": 30, "language": "ja"}

    async def transcribe(self, recording):
        self.calls.append(recording)
        return self.text


@pytest.mark.parametrize("audio", [b"invalid", wav(0.1), wav(31), wav(rate=24000), wav(channels=2), wav(width=1), wav()[:-4]])
def test_invalid_audio_rejected(audio):
    with pytest.raises(STTError):
        validate_recording(audio)


def test_valid_audio_and_silence_energy():
    recorded = validate_recording(wav(1))
    assert recorded.seconds == 1
    assert recorded.rms > 0.001
    assert validate_recording(wav(1, silent=True)).rms == 0
    with wave.open(io.BytesIO(recorded.wav), "rb") as audio:
        assert audio.getnframes() == 16000


def test_transcription_api_is_separate_from_llm_and_does_not_cache():
    service = FakeSTT()

    def forbidden(request):
        raise AssertionError("STT must not call the LLM or VOICEVOX")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
    with TestClient(create_app(client=upstream, stt=service)) as client:
        assert client.get("/api/config").json()["stt"]["available"] is True
        response = client.post("/api/stt/transcribe", content=wav(), headers={"Content-Type": "audio/wav"})
        assert response.status_code == 200
        assert response.json()["text"] == service.text
        assert response.headers["cache-control"] == "no-store"
        assert "microphone=(self)" in response.headers["permissions-policy"]
        assert len(service.calls) == 1
        assert client.app.state.sessions == {}
    asyncio.run(upstream.aclose())


@pytest.mark.parametrize(("body", "headers", "status"), [
    (b"not audio", {"Content-Type": "application/json"}, 415),
    (b"not audio", {"Content-Type": "audio/wav"}, 400),
    (b"x" * (MAX_BYTES + 1), {"Content-Type": "audio/wav"}, 413),
    (wav(), {"Content-Type": "audio/wav", "Origin": "https://foreign.example"}, 403),
])
def test_stt_request_limits_and_origin(body, headers, status):
    service = FakeSTT()
    with TestClient(create_app(stt=service)) as client:
        assert client.post("/api/stt/transcribe", content=body, headers=headers).status_code == status
        assert not client.app.state.stt_busy
        assert not service.calls


def test_stt_unavailable_and_empty_transcript():
    with TestClient(create_app(stt=FakeSTT(available=False))) as client:
        assert client.post("/api/stt/transcribe", content=wav(), headers={"Content-Type": "audio/wav"}).status_code == 503
    with TestClient(create_app(stt=FakeSTT(text=""))) as client:
        response = client.post("/api/stt/transcribe", content=wav(), headers={"Content-Type": "audio/wav"})
        assert response.json()["text"] == ""


def test_one_stt_job_at_a_time():
    entered = threading.Event()
    release = threading.Event()

    class Slow(FakeSTT):
        async def transcribe(self, recording):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return self.text

    result = []
    with TestClient(create_app(stt=Slow())) as client:
        worker = threading.Thread(target=lambda: result.append(client.post(
            "/api/stt/transcribe", content=wav(), headers={"Content-Type": "audio/wav"})))
        worker.start()
        try:
            assert entered.wait(5)
            response = client.post("/api/stt/transcribe", content=wav(), headers={"Content-Type": "audio/wav"})
            assert response.status_code == 429
        finally:
            release.set()
            worker.join(timeout=5)
        assert not worker.is_alive()
        assert result[0].status_code == 200
        assert not client.app.state.stt_busy


def test_http_disconnect_cancels_recognition_through_middleware():
    async def check():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        class Slow(FakeSTT):
            async def transcribe(self, recording):
                entered.set()
                try:
                    await asyncio.sleep(60)
                finally:
                    cancelled.set()

        app = create_app(stt=Slow())
        incoming = asyncio.Queue()
        body = wav()
        await incoming.put({"type": "http.request", "body": body, "more_body": False})
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/api/stt/transcribe", "raw_path": b"/api/stt/transcribe", "query_string": b"",
                 "headers": [(b"content-type", b"audio/wav"), (b"content-length", str(len(body)).encode())],
                 "server": ("testserver", 80), "client": ("127.0.0.1", 12345)}

        async def send(message):
            pass

        async with app.router.lifespan_context(app):
            request = asyncio.create_task(app(scope, incoming.get, send))
            try:
                await asyncio.wait_for(entered.wait(), timeout=3)
                await incoming.put({"type": "http.disconnect"})
                await asyncio.wait_for(request, timeout=3)
                assert cancelled.is_set()
                assert not app.state.stt_busy
            finally:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
    asyncio.run(check())


class FakeInput:
    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, data):
        self.data += data

    async def drain(self):
        pass

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, output=None, code=0):
        self.stdin = FakeInput()
        self.stdout = asyncio.StreamReader()
        self.returncode = code if output is not None else None
        self.finished = asyncio.Event()
        self.killed = False
        self.waited = False
        if output is not None:
            self.stdout.feed_data(output)
            self.stdout.feed_eof()
            self.finished.set()

    async def wait(self):
        await self.finished.wait()
        self.waited = True
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self.finished.set()


def service_ready(monkeypatch, timeout=1):
    service = WhisperSTT(STTConfig(model_path="/test/model.bin", timeout=timeout))
    monkeypatch.setattr(service, "status", lambda: {"available": True})
    return service


def test_cli_uses_stdin_stdout_and_no_shell(monkeypatch):
    async def check():
        process = FakeProcess("こんにちは。\n".encode())
        async def spawn(*args, **kwargs):
            assert args[args.index("--file") + 1] == "-"
            assert args[args.index("--output-file") + 1] == "-"
            assert "--output-txt" in args  # Required for stdout with stdin input in v1.9.5.
            assert "--no-gpu" in args
            assert args[args.index("--language") + 1] == "ja"
            assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        service = service_ready(monkeypatch)
        recording = validate_recording(wav())
        assert await service.transcribe(recording) == "こんにちは。"
        assert process.stdin.data == recording.wav
        assert process.stdin.closed and process.waited
    asyncio.run(check())


def test_silence_does_not_start_recognizer(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("silence must not invoke Whisper")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    service = service_ready(monkeypatch)
    assert asyncio.run(service.transcribe(validate_recording(wav(silent=True)))) == ""


def test_cancellation_during_process_creation_is_reaped(monkeypatch):
    async def check():
        process = FakeProcess()
        spawning = asyncio.Event()

        async def spawn(*args, **kwargs):
            spawning.set()
            await asyncio.sleep(0.02)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(service_ready(monkeypatch).transcribe(validate_recording(wav())))
        await spawning.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.killed and process.waited
    asyncio.run(check())


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_kill_child(monkeypatch, cancel):
    async def check():
        process = FakeProcess()
        spawned = asyncio.Event()
        async def spawn(*args, **kwargs):
            spawned.set()
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        service = service_ready(monkeypatch, timeout=0.03 if not cancel else 5)
        task = asyncio.create_task(service.transcribe(validate_recording(wav())))
        await spawned.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(STTError) as error:
                await task
            assert error.value.status == 504
        assert process.killed and process.waited
    asyncio.run(check())


@pytest.mark.parametrize(("output", "code", "status"), [(b"secret", 1, 502), (b"x" * 65537, 0, 422), (b"x" * 4001, 0, 422), (b"\xff", 0, 502)])
def test_native_errors_and_output_limits_are_safe(monkeypatch, output, code, status):
    async def check():
        async def spawn(*args, **kwargs):
            return FakeProcess(output, code)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        with pytest.raises(STTError) as error:
            await service_ready(monkeypatch).transcribe(validate_recording(wav()))
        assert error.value.status == status
        assert "secret" not in str(error.value)
    asyncio.run(check())
