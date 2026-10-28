"""Per-conversation backend selection and bounded, text-only chat history."""

import ipaddress
from dataclasses import replace
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .providers import Settings


def backend_url(value: str, *, llm=False) -> str:
    value = value.strip()
    if any(ord(char) < 33 for char in value) or "\\" in value:
        raise ValueError("URL に空白や制御文字は使えません。")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("HTTP(S) のベース URL を指定してください。認証情報・クエリ・フラグメントは使えません。")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("URL のポート番号が不正です。")
    # Private/loopback endpoints are explicitly needed for LAN and local VOICEVOX.
    # This is not a network sandbox: only trusted LAN clients should access this app.
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address and (address.is_link_local or address.is_multicast or address.is_unspecified):
        raise ValueError("リンクローカル・マルチキャスト・未指定アドレスは接続先にできません。")
    if parsed.hostname.lower().rstrip(".") == "metadata.google.internal":
        raise ValueError("この接続先は利用できません。")
    path = parsed.path.rstrip("/")
    if llm:
        path = path.removesuffix("/chat/completions") or "/v1"
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port and (parsed.scheme, port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    return urlunsplit((parsed.scheme, host, path, "", ""))


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    llm_url: str = Field(max_length=2048)
    llm_model: str = Field(min_length=1, max_length=200)
    voicevox_url: str = Field(max_length=2048)
    # Never stored by the server or echoed in responses. None can use a matching server key.
    llm_api_key: str | None = Field(default=None, max_length=2048)
    reasoning_effort: Literal["", "none", "minimal", "low", "medium", "high"] = ""

    @field_validator("llm_url")
    @classmethod
    def normalize_llm(cls, value):
        return backend_url(value, llm=True)

    @field_validator("voicevox_url")
    @classmethod
    def normalize_voicevox(cls, value):
        return backend_url(value)

    @field_validator("llm_model")
    @classmethod
    def nonempty_model(cls, value):
        if not value.strip():
            raise ValueError("モデル名を指定してください。")
        return value.strip()

    def resolve(self, server: Settings) -> Settings:
        key = self.llm_api_key
        if key is None:
            # Never forward server-held credentials to a browser-selected different backend.
            try:
                same_backend = self.llm_url == backend_url(server.llm_base_url, llm=True)
            except ValueError:
                same_backend = False
            key = server.llm_api_key if same_backend else ""
        return replace(server, llm_base_url=self.llm_url, llm_model=self.llm_model,
                       voicevox_url=self.voicevox_url, llm_api_key=key,
                       llm_reasoning_effort=self.reasoning_effort)


class SpeakerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    voicevox_url: str = Field(max_length=2048)

    @field_validator("voicevox_url")
    @classmethod
    def normalize(cls, value):
        return backend_url(value)
