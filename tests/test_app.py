import asyncio
import base64
import io
import json
import threading
import wave

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from restream.app import create_app
from restream.providers import ProviderError, Settings, llm_stream, synthesize


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 240)
    return buffer.getvalue()


def engine(request):
    if request.url.path == "/speakers":
        return httpx.Response(200, json=[{"name": "Test", "styles": [{"name": "Normal", "id": 3}]}])
    if request.url.path == "/audio_query":
        assert request.url.params["speaker"] == "3"
        return httpx.Response(200, json={"speedScale": 1, "accent_phrases": []})
    if request.url.path == "/synthesis":
        query = json.loads(request.content)
        assert query["outputSamplingRate"] == 24000
        assert query["outputStereo"] is False
        return httpx.Response(200, content=wav_bytes())
    raise AssertionError(request.url)


@pytest.fixture
def client():
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    with TestClient(create_app(Settings(), upstream)) as test:
        yield test
    asyncio.run(upstream.aclose())


def drain(ws, acknowledge=True):
    events = []
    for _ in range(1000):
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "audio" and acknowledge:
            assert base64.b64decode(event["wav_base64"]) == wav_bytes()
            ws.send_json({"type": "played", "seq": event["seq"]})
        if event["type"] in {"done", "error", "cancelled"}:
            return events
    raise AssertionError("session did not terminate")


def test_static_and_config(client):
    assert client.get("/").status_code == 200
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/api/config").json()["auth"] is False
    assert client.get("/api/speakers").json()[0]["name"] == "Test"


@pytest.mark.parametrize("audio", [False, True])
def test_text_stream_end_to_end(client, audio):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "audio": audio})
        assert ws.receive_json()["type"] == "ready"
        chunks = ["こんにちは", "。これは", "テストです。", "末尾"]
        for chunk in chunks:
            ws.send_json({"type": "text", "text": chunk})
        ws.send_json({"type": "end"})
        events = drain(ws)
        assert events[-1]["type"] == "done"
        assert "".join(e["text"] for e in events if e["type"] == "segment") == "".join(chunks)
        ids = [e["seq"] for e in events if e["type"] == "audio"]
        assert ids == ([1, 2, 3] if audio else [])


def test_empty_and_whitespace_input(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        ws.send_json({"type": "text", "text": " \n "})
        ws.send_json({"type": "end"})
        events = drain(ws)
        assert not any(e["type"] == "audio" for e in events)
        assert events[-1]["type"] == "done"


def test_external_app_to_browser(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "source": "external"})
        ready = ws.receive_json()
        path = f"/api/sessions/{ready['session_id']}/events"
        assert client.post(path, json={"type": "text", "text": "外部アプリから"}).status_code == 202
        assert client.post(path, json={"type": "text", "text": "送信しました。"}).status_code == 202
        assert client.post(path, json={"type": "end"}).status_code == 202
        events = drain(ws)
        assert events[-1]["type"] == "done"
        assert "".join(e["text"] for e in events if e["type"] == "segment") == "外部アプリから送信しました。"
        assert any(e["type"] == "audio" for e in events)
    assert client.post(path, json={"type": "end"}).status_code == 404


def test_external_input_validation_and_origin(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "source": "external", "audio": False})
        ready = ws.receive_json()
        path = f"/api/sessions/{ready['session_id']}/events"
        assert client.post(path, json={"type": "text", "text": "x"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post(path, json={"type": "text", "text": "x" * 8193}).status_code == 422
        assert client.post(path, content=b"x" * 65537).status_code == 413
        assert client.post(path, json={"type": "end"}).status_code == 202
        assert drain(ws)[-1]["type"] == "done"


def test_external_cannot_inject_into_normal_session(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "audio": False})
        ready = ws.receive_json()
        path = f"/api/sessions/{ready['session_id']}/events"
        assert client.post(path, json={"type": "text", "text": "injected"}).status_code == 404
        ws.send_json({"type": "end"})
        assert drain(ws)[-1]["type"] == "done"


def test_external_backpressure_does_not_accept_end_when_full(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "source": "external", "mode": "raw"})
        ready = ws.receive_json()
        path = f"/api/sessions/{ready['session_id']}/events"
        for _ in range(100):
            response = client.post(path, json={"type": "text", "text": "待機中"})
            if response.status_code == 429:
                break
            assert response.status_code == 202
        else:
            raise AssertionError("input was not bounded")
        # No playback ACKs: rendering is blocked and the queue remains full.
        assert client.post(path, json={"type": "end"}).status_code == 429
        assert not client.app.state.sessions[ready["session_id"]].ended
        ws.send_json({"type": "cancel"})
        assert drain(ws, acknowledge=False)[-1]["type"] == "cancelled"


def test_cancel_and_new_session(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["type"] == "cancelled"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "audio": False})
        ws.send_json({"type": "end"})
        assert drain(ws)[-1]["type"] == "done"


def test_audio_window_cancel_without_ack(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "mode": "raw"})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(6):
            ws.send_json({"type": "text", "text": "こんにちは"})
        ws.send_json({"type": "end"})
        count = 0
        while count < 3:
            event = ws.receive_json()
            count += event["type"] == "audio"
        ws.send_json({"type": "cancel"})
        events = drain(ws, acknowledge=False)
        assert not any(e["type"] == "audio" for e in events)
        assert events[-1]["type"] == "cancelled"


@pytest.mark.parametrize("event", [
    {"type": "start", "mode": "unknown"},
    {"type": "start", "speaker": "3"},
    {"type": "start", "min_chars": 200, "max_chars": 100},
    {"type": "start", "source": "llm"},
])
def test_invalid_start(client, event):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(event)
        assert ws.receive_json()["type"] == "error"


@pytest.mark.parametrize("event", [[], {"type": "played", "seq": 1}, {"type": "text", "text": "a" * 8193}, {"type": "nope"}])
def test_invalid_events(client, event):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "audio": False})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(event)
        assert ws.receive_json()["type"] == "error"


def test_origin_check(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws", headers={"Origin": "https://evil.example"}):
        pass
    with client.websocket_connect("/ws", headers={"Origin": "http://testserver"}) as ws:
        ws.send_json({"type": "start", "audio": False})
        ws.send_json({"type": "end"})
        assert drain(ws)[-1]["type"] == "done"


def test_session_limit_and_disconnect_cleanup():
    with TestClient(create_app(Settings(max_sessions=1))) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start", "audio": False})
            assert ws.receive_json()["type"] == "ready"
            with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws"):
                pass
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start", "audio": False})
            ws.send_json({"type": "end"})
            assert drain(ws)[-1]["type"] == "done"


def sse(text):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]}, ensure_ascii=False) + "\n\n"


def test_llm_integration():
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer private-key"
        assert json.loads(request.content)["model"] == "test-model"
        return httpx.Response(200, text=": heartbeat\n\n" + sse("こんにちは。") + sse("続きです") + "data: [DONE]\n\n")
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(create_app(Settings(llm_model="test-model", llm_api_key="private-key"), upstream)) as client:
        assert "private-key" not in client.get("/api/config").text
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start", "source": "llm", "prompt": "挨拶して", "audio": False})
            events = drain(ws)
            assert events[-1]["type"] == "done"
            assert "".join(e["text"] for e in events if e["type"] == "segment") == "こんにちは。続きです"
    asyncio.run(upstream.aclose())


@pytest.mark.parametrize("effort", ["", "none", "medium"])
def test_reasoning_effort_opt_in_and_reasoning_not_spoken(effort):
    def handler(request):
        payload = json.loads(request.content)
        if effort:
            assert payload["reasoning_effort"] == effort
        else:
            assert "reasoning_effort" not in payload
        reasoning = {"choices": [{"delta": {"reasoning": "private reasoning", "reasoning_content": "hidden"}}]}
        body = "data: " + json.dumps(reasoning) + "\n\n" + sse("回答です。") + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            chunks = [text async for text in llm_stream(
                client, Settings(llm_model="test", llm_reasoning_effort=effort), "prompt")]
            assert "".join(chunks) == "回答です。"
    asyncio.run(check())


def test_reasoning_effort_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    assert Settings.from_env().llm_reasoning_effort == "none"


@pytest.mark.parametrize("body", [sse("途中"), "data: invalid-json\n\n", "data: {\"error\":\"secret\"}\n\n"])
def test_llm_truncation_or_bad_stream(body):
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))) as client:
            with pytest.raises(ProviderError) as error:
                _ = [text async for text in llm_stream(client, Settings(llm_model="test"), "prompt")]
            assert "secret" not in str(error.value)
    asyncio.run(check())


def test_voicevox_failure_redacted():
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500, text="secret"))) as client:
            with pytest.raises(ProviderError) as error:
                await synthesize(client, Settings(), "hello", 3, 1.0)
            assert "secret" not in str(error.value)
    asyncio.run(check())


def test_engine_failure_propagates_to_websocket():
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    with TestClient(create_app(Settings(), upstream)) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        ws.send_json({"type": "text", "text": "こんにちは。"})
        ws.send_json({"type": "end"})
        assert drain(ws)[-1]["type"] == "error"
    asyncio.run(upstream.aclose())


def test_disconnect_cancels_inflight_synthesis():
    started = threading.Event()
    cancelled = threading.Event()

    async def handler(request):
        if request.url.path == "/audio_query":
            return httpx.Response(200, json={})
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return httpx.Response(200, content=wav_bytes())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(create_app(Settings(), upstream)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start"})
            ws.send_json({"type": "text", "text": "こんにちは。"})
            ws.send_json({"type": "end"})
            assert started.wait(timeout=5)
        assert cancelled.wait(timeout=5)
    asyncio.run(upstream.aclose())


def test_multiline_sse_and_utf8_transport_fragments():
    class BytesStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            payload = ('data: {"choices":\r\n'
                       'data: [{"delta":{"content":"日本語"},"finish_reason":"stop"}]}\r\n\r\n').encode()
            for byte in payload:
                yield bytes([byte])

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=BytesStream(), headers={"Content-Type": "text/event-stream"})
        )) as client:
            chunks = [text async for text in llm_stream(client, Settings(llm_model="test"), "prompt")]
            assert chunks == ["日本語"]
    asyncio.run(check())
