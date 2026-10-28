import asyncio
import io
import json
import wave

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from restream.app import Start, create_app
from restream.chat import BackendConfig, backend_url
from restream.providers import Settings

BACKEND = {"llm_url": "https://selected.example", "llm_model": "selected-model",
           "voicevox_url": "http://192.168.1.20:50021", "reasoning_effort": "none"}


def done(ws):
    events = []
    for _ in range(100):
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "audio":
            ws.send_json({"type": "played", "seq": event["seq"]})
        if event["type"] in {"done", "error"}:
            return events
    raise AssertionError("session did not finish")


@pytest.mark.parametrize(("url", "expected"), [
    ("https://OLLAMA.example/", "https://ollama.example/v1"),
    ("https://ollama.example:443/v1/", "https://ollama.example/v1"),
    ("http://localhost:11434/v1/chat/completions", "http://localhost:11434/v1"),
    ("http://[::1]:11434", "http://[::1]:11434/v1"),
    ("https://proxy.example/llm/v1", "https://proxy.example/llm/v1"),
])
def test_normalize_backend(url, expected):
    assert backend_url(url, llm=True) == expected


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com", "https://user:secret@example.com",
    "https://example.com?token=secret", "https://example.com#fragment", "https://",
    "http://example.com:70000", "http://169.254.169.254", "http://0.0.0.0",
    "http://[fe80::1]", "http://224.0.0.1", "http://metadata.google.internal",
    "http://example.com/\rheader", "http://example.com\\@evil.example",
])
def test_invalid_backend(url):
    with pytest.raises(ValueError):
        backend_url(url)


def test_server_key_is_not_forwarded_to_changed_destination():
    server = Settings(llm_base_url="https://original.example/v1", llm_api_key="server-secret")
    selected = BackendConfig(**BACKEND).resolve(server)
    assert selected.llm_api_key == ""
    assert server.llm_base_url == "https://original.example/v1"
    same = BackendConfig(**{**BACKEND, "llm_url": "https://ORIGINAL.example:443/v1/"}).resolve(server)
    assert same.llm_api_key == "server-secret"
    explicit = BackendConfig(**{**BACKEND, "llm_api_key": "provided-key"}).resolve(server)
    assert explicit.llm_api_key == "provided-key"


@pytest.mark.parametrize("messages", [
    [], [{"role": "assistant", "content": "wrong"}],
    [{"role": "system", "content": "only system"}],
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "tool", "content": "not supported"}],
    [{"role": "user", "content": "a" * 16001}],
    [{"role": "user", "content": "a" * 8000}, {"role": "assistant", "content": "b" * 8000},
     {"role": "user", "content": "c"}],
])
def test_invalid_history(messages):
    with pytest.raises(ValidationError):
        Start(type="start", source="llm", messages=messages)


def test_prompt_and_history_are_mutually_exclusive():
    with pytest.raises(ValidationError):
        Start(type="start", source="llm", prompt="x", messages=[{"role": "user", "content": "x"}])


def test_chat_history_custom_backends_and_audio():
    requests = []
    history = [{"role": "system", "content": "日本語で答えて"},
               {"role": "user", "content": "私の好きな色は青です。"},
               {"role": "assistant", "content": "青ですね。"},
               {"role": "user", "content": "好きな色は何でしたか？"}]

    def handler(request):
        requests.append(request)
        if request.url.host == "selected.example":
            assert request.url.path == "/v1/chat/completions"
            assert "authorization" not in request.headers
            payload = json.loads(request.content)
            assert payload["model"] == "selected-model"
            assert payload["messages"] == history
            assert payload["reasoning_effort"] == "none"
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"青です。"}}]}\n\ndata: [DONE]\n\n')
        assert request.url.host == "192.168.1.20"
        if request.url.path == "/audio_query":
            return httpx.Response(200, json={})
        assert request.url.path == "/synthesis"
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x01\x00" * 240)
        return httpx.Response(200, content=audio.getvalue())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    server = Settings(llm_base_url="https://original.example/v1", llm_api_key="server-secret")
    with TestClient(create_app(server, upstream)) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "source": "llm", "messages": history, "backend": BACKEND})
        events = done(ws)
        assert events[-1]["type"] == "done"
        assert "".join(e["text"] for e in events if e["type"] == "delta") == "青です。"
        assert sum(e["type"] == "audio" for e in events) == 1
    assert len(requests) == 3
    asyncio.run(upstream.aclose())


def test_session_selected_backend_does_not_mutate_server_default():
    models = []

    def handler(request):
        models.append(json.loads(request.content)["model"])
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"返答。"}}]}\n\ndata: [DONE]\n\n')

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(create_app(Settings(llm_model="server-default"), upstream)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start", "source": "llm", "prompt": "a", "audio": False, "backend": BACKEND})
            assert done(ws)[-1]["type"] == "done"
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start", "source": "llm", "prompt": "b", "audio": False})
            assert done(ws)[-1]["type"] == "done"
    assert models == ["selected-model", "server-default"]
    asyncio.run(upstream.aclose())


def test_ui_routes_and_custom_speaker_discovery():
    def handler(request):
        assert request.url == "http://192.168.1.20:50021/speakers"
        return httpx.Response(200, json=[{"name": "ずんだもん", "styles": [{"id": 3, "name": "ノーマル"}]}])

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(create_app(Settings(llm_api_key="secret"), upstream)) as client:
        assert "VOICEVOX Chat" in client.get("/").text
        assert "Restream Lab" in client.get("/lab").text
        assert client.get("/static/chat.js").status_code == 200
        assert "secret" not in client.get("/api/config").text
        response = client.post("/api/backends/speakers", json={"voicevox_url": BACKEND["voicevox_url"]})
        assert response.status_code == 200
        assert response.json()[0]["name"] == "ずんだもん"
        assert client.post("/api/backends/speakers", json={"voicevox_url": "file:///tmp"}).status_code == 422
        assert client.post("/api/backends/speakers", content=b"x" * 8193).status_code == 413
        assert client.post("/api/backends/speakers", json={"voicevox_url": BACKEND["voicevox_url"]},
                           headers={"Origin": "https://foreign.example"}).status_code == 403
    asyncio.run(upstream.aclose())


def test_config_does_not_expose_url_credentials():
    with TestClient(create_app(Settings(llm_base_url="https://user:secret@example.com/v1",
                                        voicevox_url="http://local.example?token=secret"))) as client:
        response = client.get("/api/config")
        assert "secret" not in response.text
        assert response.json()["llm_url"] == ""
        assert response.json()["voicevox_url"] == ""
