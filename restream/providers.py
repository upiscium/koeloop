import json
import os
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from .pauses import PauseSettings, append_sentence_silence, apply_pauses


class ProviderError(Exception):
    """Safe, user-facing provider failure (never includes response bodies or keys)."""


@dataclass(frozen=True)
class Settings:
    voicevox_url: str = "http://127.0.0.1:50021"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_reasoning_effort: str = ""
    max_sessions: int = 4

    @classmethod
    def from_env(cls):
        return cls(
            voicevox_url=os.getenv("VOICEVOX_URL", cls.voicevox_url).rstrip("/"),
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url).rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", ""),
            max_sessions=max(1, int(os.getenv("MAX_SESSIONS", "4"))),
        )


async def llm_stream(client: httpx.AsyncClient, settings: Settings, prompt: str,
                     messages: list[dict[str, str]] | None = None) -> AsyncIterator[str]:
    if not settings.llm_model:
        raise ProviderError("LLM_MODEL をサーバーに設定してください。")
    headers = {"Accept": "text/event-stream"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload = {"model": settings.llm_model,
               "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
               "stream": True}
    # Opt-in only: compatibility and supported effort names differ between providers.
    if settings.llm_reasoning_effort:
        payload["reasoning_effort"] = settings.llm_reasoning_effort
    try:
        async with client.stream("POST", f"{settings.llm_base_url}/chat/completions",
                                 headers=headers, json=payload) as response:
            response.raise_for_status()
            data = []
            size = 0
            finished = False
            async for line in response.aiter_lines():
                # SSE events may span several data: lines; transport chunks are irrelevant.
                if line == "":
                    if not data:
                        continue
                    raw = "\n".join(data)
                    data = []
                    size = 0
                    if raw.strip() == "[DONE]":
                        return
                    event = json.loads(raw)
                    if not isinstance(event, dict) or "error" in event:
                        raise ProviderError("LLM がエラーイベントを返しました。")
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) != 0:
                            continue
                        content = choice.get("delta", {}).get("content")
                        if isinstance(content, str) and content:
                            yield content
                        if choice.get("finish_reason") is not None:
                            finished = True
                    continue
                if line.startswith("data:"):
                    value = line[5:].removeprefix(" ")
                    size += len(value)
                    if size > 65536:
                        raise ProviderError("LLM のイベントが上限を超えました。")
                    data.append(value)
            if data or not finished:
                raise ProviderError("LLM のストリームが完了通知なしで切断されました。")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        raise ProviderError("LLM 接続またはストリーム形式に問題があります。サーバー設定を確認してください。") from exc


async def synthesize(client: httpx.AsyncClient, settings: Settings, text: str, speaker: int, speed: float,
                     pauses: PauseSettings | None = None) -> bytes:
    try:
        query_response = await client.post(f"{settings.voicevox_url}/audio_query",
                                          params={"text": text, "speaker": speaker})
        query_response.raise_for_status()
        query = query_response.json()
        query["speedScale"] = speed
        query["outputSamplingRate"] = 24000
        query["outputStereo"] = False
        # Omitted settings preserve the Engine baseline used by earlier comparisons.
        if pauses is not None:
            apply_pauses(query, text, pauses)
        response = await client.post(f"{settings.voicevox_url}/synthesis",
                                     params={"speaker": speaker}, json=query)
        response.raise_for_status()
        audio = response.content
        if len(audio) > 16 * 1024 * 1024 or not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
            raise ProviderError("VOICEVOX が有効な WAV を返しませんでした。")
        if pauses is not None:
            try:
                return append_sentence_silence(audio, text, pauses)
            except (wave.Error, ValueError, EOFError) as exc:
                raise ProviderError("句点の間を調整できない WAV 形式です。句読点の間の調整をオフにして試してください。") from exc
        return audio
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ProviderError("VOICEVOX の合成に失敗しました。Engine と話者 ID を確認してください。") from exc
