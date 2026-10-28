import pytest
from fastapi.testclient import TestClient

from restream.app import create_app


@pytest.mark.parametrize(("payload", "expected"), [
    ({"type": "start", "speed": 0}, "speed"),
    ({"type": "start", "speed": "1.2"}, "0.5～2.0"),
    ({"type": "start", "speaker": -1}, "speaker"),
    ({"type": "start", "source": "llm", "prompt": "あ" * 16001}, "16000"),
    ({"type": "start", "mode": "not-a-mode"}, "mode"),
    ({"type": "start", "comma_pause_scale": -1}, "comma_pause_scale"),
])
def test_start_errors_identify_invalid_setting(payload, expected):
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_json(payload)
        event = ws.receive_json()
        assert event["type"] == "error"
        assert expected in event["message"]


def test_unknown_field_suggests_restart_without_echoing_field_or_value():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "private-field-name": "private-value"})
        event = ws.receive_json()
        assert "再起動" in event["message"]
        assert "private" not in event["message"]


def test_invalid_backend_does_not_echo_credentials_or_prompt():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "source": "llm", "prompt": "private-prompt",
                      "backend": {"llm_url": "https://user:private-url-secret@example.com",
                                  "llm_model": "test", "voicevox_url": "http://127.0.0.1:50021",
                                  "llm_api_key": "private-key"}})
        event = ws.receive_json()
        assert "backend" in event["message"]
        assert "private" not in event["message"]


def test_invalid_json_does_not_echo_partial_secrets():
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_text('{"type":"start","private-key":"secret')
        event = ws.receive_json()
        assert "JSON" in event["message"]
        assert "private" not in event["message"]
        assert "secret" not in event["message"]


def test_ui_assets_are_revalidated():
    with TestClient(create_app()) as client:
        for path in ("/", "/lab", "/static/app.js", "/static/chat.js"):
            assert client.get(path).headers["cache-control"] == "no-cache"
