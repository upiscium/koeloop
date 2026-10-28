import asyncio
import base64
import io
import json
import sys
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from restream import vosk_stream_worker
from restream.app import create_app
from restream.vad import ResidentSilero, VADStart, native_call


class FakeVoice:
    def __init__(self):
        self.starts = 0
        self.closed = 0
        self.pcm = []
        self.endpoint = 3
        self.speech = True

    def status(self):
        return {"available": True, "streaming": True}

    async def start(self, settings):
        self.starts += 1
        self.pcm = []
        return self

    def feed(self, pcm):
        self.pcm.append(pcm)
        return self.end() if len(self.pcm) % self.endpoint == 0 else {"type": "state", "state": "speaking"}

    def end(self):
        frames = sum(len(pcm) // 2 for pcm in self.pcm) if self.speech else 0
        return {"type": "end", "speech": frames > 0, "start_sample": 0, "end_sample": frames}

    async def close(self):
        self.closed += 1


def test_state_ack_then_endpoint_without_any_transcript_or_llm():
    service = FakeVoice()
    with TestClient(create_app(voice_vad=service)) as client:
        assert client.get("/api/config").json()["voice_vad"]["available"]
        with client.websocket_connect("/ws/vad") as ws:
            ws.send_json({"type": "start"})
            assert ws.receive_json()["type"] == "ready"
            for index in range(3):
                ws.send_bytes(b"\x01\x00" * 1600)
                event = ws.receive_json()
                if index == 0:
                    assert event == {"type": "state", "state": "speaking"}
                    event = ws.receive_json()
                if index < 2:
                    assert event == {"type": "ack", "seq": index + 1}
                else:
                    assert event == {"type": "end", "speech": True, "reason": "endpoint", "start_sample": 0, "end_sample": 4800}
        assert client.app.state.sessions == {}
        assert not client.app.state.stt_busy
        assert service.closed == 0  # Normal completion keeps a model resident.
    assert service.closed == 1


def test_manual_end_can_return_empty_without_submission():
    service = FakeVoice()
    service.speech = False
    with TestClient(create_app(voice_vad=service)) as client, client.websocket_connect("/ws/vad") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "end"})
        assert ws.receive_json() == {"type": "end", "speech": False, "reason": "manual", "start_sample": 0, "end_sample": 0}


@pytest.mark.parametrize("command", ["cancel", "disconnect"])
def test_abort_releases_slot_and_worker(command):
    service = FakeVoice()
    with TestClient(create_app(voice_vad=service)) as client:
        with client.websocket_connect("/ws/vad") as ws:
            ws.send_json({"type": "start"})
            assert ws.receive_json()["type"] == "ready"
            if command == "cancel":
                ws.send_json({"type": "cancel"})
                assert ws.receive_json()["type"] == "cancelled"
        assert not client.app.state.stt_busy
        assert service.closed == 0  # Shared graph is stateless; only the turn is discarded.
        with client.websocket_connect("/ws/vad") as ws:
            ws.send_json({"type": "start"})
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "end"})
            assert ws.receive_json()["type"] == "end"


def test_origin_and_stt_admission_limits():
    with TestClient(create_app(voice_vad=FakeVoice())) as client:
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/vad", headers={"Origin": "https://evil.example"}):
            pass
        with client.websocket_connect("/ws/vad", headers={"Origin": "http://testserver"}) as ws:
            ws.send_json({"type": "start"})
            assert ws.receive_json()["type"] == "ready"
            with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/vad"):
                pass
            ws.send_json({"type": "cancel"})
            assert ws.receive_json()["type"] == "cancelled"


@pytest.mark.parametrize("first", ['{"type":"start","model":"untrusted"}', '[]', 'not-json', 'x' * 129])
def test_start_cannot_override_model_or_paths(first):
    with TestClient(create_app(voice_vad=FakeVoice())) as client, client.websocket_connect("/ws/vad") as ws:
        ws.send_text(first)
        assert ws.receive_json()["type"] == "error"


@pytest.mark.parametrize("initial", [{"type": "start"}, {"type": "invalid"}])
def test_idle_handshake_does_not_reserve_or_release_another_sessions_slot(initial):
    with TestClient(create_app(voice_vad=FakeVoice())) as client:
        with client.websocket_connect("/ws/vad") as idle, client.websocket_connect("/ws/vad") as active:
            active.send_json({"type": "start"})
            assert active.receive_json()["type"] == "ready"
            idle.send_json(initial)
            if initial["type"] == "start":
                with pytest.raises(WebSocketDisconnect):
                    idle.receive_json()
            else:
                assert idle.receive_json()["type"] == "error"
                assert idle.receive()["type"] == "websocket.close"
            assert client.app.state.stt_busy
            active.send_json({"type": "cancel"})
            assert active.receive_json()["type"] == "cancelled"
        assert not client.app.state.stt_busy


@pytest.mark.parametrize("stage", ["loading", "inference"])
def test_cancel_ack_is_prompt_but_slot_is_held_until_native_thread_exits(stage):
    entered, release = Event(), Event()

    def block():
        entered.set()
        assert release.wait(10), "test did not release the native thread"

    class SlowNative(FakeVoice):
        async def start(self, settings):
            if stage == "loading":
                await native_call(block)
            return await super().start(settings)

        def feed(self, pcm):
            block()
            return super().feed(pcm)

    try:
        with TestClient(create_app(voice_vad=SlowNative())) as client:
            with client.websocket_connect("/ws/vad") as ws:
                ws.send_json({"type": "start"})
                if stage == "inference":
                    assert ws.receive_json()["type"] == "ready"
                    ws.send_bytes(b"\x01\x00" * 1600)
                assert entered.wait(5)
                ws.send_json({"type": "cancel"})
                assert ws.receive_json()["type"] == "cancelled"
                assert client.app.state.stt_busy
                with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/vad"):
                    pass
                release.set()
                assert ws.receive()["type"] == "websocket.close"
            assert not client.app.state.stt_busy
    finally:
        release.set()


@pytest.mark.parametrize("data", [b"", b"1", b"x" * 6402])
def test_invalid_pcm_size_is_rejected(data):
    with TestClient(create_app(voice_vad=FakeVoice())) as client, client.websocket_connect("/ws/vad") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(data)
        assert ws.receive_json()["type"] == "error"


def test_cancel_while_loading_does_not_wait_for_model():
    class Slow(FakeVoice):
        async def start(self, settings):
            await asyncio.sleep(60)

    service = Slow()
    with TestClient(create_app(voice_vad=service)) as client, client.websocket_connect("/ws/vad") as ws:
        ws.send_json({"type": "start"})
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["type"] == "cancelled"


def test_error_never_echoes_native_transcript_or_paths():
    class Broken(FakeVoice):
        def feed(self, pcm):
            raise RuntimeError("secret transcript /private/model")

    with TestClient(create_app(voice_vad=Broken())) as client, client.websocket_connect("/ws/vad") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"\x01\x00" * 1600)
        event = ws.receive_json()
        assert event["type"] == "error"
        assert "secret" not in json.dumps(event)


def test_input_queue_is_bounded_while_model_is_loading():
    class Slow(FakeVoice):
        async def start(self, settings):
            await asyncio.sleep(60)

    with TestClient(create_app(voice_vad=Slow())) as client, client.websocket_connect("/ws/vad") as ws:
        ws.send_json({"type": "start"})
        for _ in range(21):
            ws.send_bytes(b"\x00\x00" * 1600)
        event = ws.receive_json()
        assert event["type"] == "error"
        assert "追いつきません" in event["message"]


def test_audio_duration_is_bounded_even_for_a_fast_sender():
    service = FakeVoice()
    service.endpoint = 1000
    with TestClient(create_app(voice_vad=service)) as client, client.websocket_connect("/ws/vad") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"
        for index in range(150):
            ws.send_bytes(b"\x01\x00" * 3200)
            event = ws.receive_json()
            if index == 0:
                assert event["type"] == "state"
                event = ws.receive_json()
            assert event["type"] == "ack"
        ws.send_bytes(b"\x01\x00")
        assert "30秒" in ws.receive_json()["message"]


def test_worker_reuses_model_but_not_recognizer_and_ignores_silent_endpoint(monkeypatch):
    models, recognizers = [], []

    def model(*, model_path):
        models.append(model_path)
        return "shared-model"

    class Recognizer:
        def __init__(self, model, rate):
            assert model == "shared-model" and rate == 16000
            recognizers.append(self)

        def AcceptWaveform(self, pcm):
            return True

        def Result(self):
            return json.dumps({"text": "3本買います。"})

        def FinalResult(self):
            return json.dumps({"text": "3本買います。"})

    def audio(pcm):
        return {"type": "audio", "pcm": base64.b64encode(pcm).decode()}

    commands = [{"type": "start"}, audio(b"\x00\x00" * 1600), {"type": "end"},
                {"type": "start"}, audio(b"\x20\x20" * 1600)]
    incoming = b"".join((json.dumps(event) + "\n").encode() for event in commands)
    output = io.StringIO()
    monkeypatch.setitem(sys.modules, "vosk", SimpleNamespace(Model=model, KaldiRecognizer=Recognizer, SetLogLevel=lambda _: None))
    monkeypatch.setattr(sys, "argv", ["worker", "--model-dir", "/fixed/model"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(incoming)))
    monkeypatch.setattr(sys, "stdout", output)
    vosk_stream_worker.main()
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert models == ["/fixed/model"] and len(recognizers) == 2
    assert events == [{"type": "loaded"}, {"type": "ready"}, {"type": "partial", "text": ""},
                      {"type": "final", "text": ""}, {"type": "ready"}, {"type": "final", "text": "3本買います。"}]


def test_resident_model_reuses_graph_but_never_shares_recurrent_state(monkeypatch):
    np = pytest.importorskip("numpy")
    loaded = []

    class Session:
        def __init__(self, path, **kwargs):
            loaded.append(path)
            assert kwargs["providers"] == ["CPUExecutionProvider"]

        def run(self, outputs, inputs):
            assert inputs["input"].shape == (1, 576)
            assert inputs["sr"] == 16000
            return np.array([[0.9]], dtype=np.float32), inputs["state"] + 1

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=Session, disable_telemetry_events=lambda: None))

    async def check():
        service = ResidentSilero("/fixed/vad.onnx")
        monkeypatch.setattr(service, "status", lambda: {"available": True})
        first = await service.start(VADStart(type="start"))
        first.feed(b"\x01\x00" * 512)
        second = await service.start(VADStart(type="start"))
        assert loaded == ["/fixed/vad.onnx"]
        assert first.scorer.session is second.scorer.session
        assert np.all(first.scorer.state == 1)
        assert np.all(second.scorer.state == 0)
        await service.close()
        assert service.closed and service.session is None
    asyncio.run(check())
