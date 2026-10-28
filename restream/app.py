import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.requests import ClientDisconnect

from .chat import BackendConfig, ChatMessage, SpeakerRequest, backend_url
from .pauses import PauseSettings
from .providers import ProviderError, Settings, llm_stream, synthesize
from .segmenter import Segmenter
from .stt import BACKENDS, MAX_BYTES, STTError, create_stt, validate_recording
from .syntax import MODEL
from .syntax import installed as syntax_installed
from .vad import ResidentSilero
from .validation import start_error_message
from .voice_stream import voice_websocket

STATIC = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)


class Start(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["start"]
    source: Literal["text", "llm", "external"] = "text"
    mode: Literal["raw", "sentence", "balanced", "syntax"] = "balanced"
    audio: bool = True
    speaker: int = Field(default=3, ge=0, le=100000)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    adjust_pauses: bool = False
    sentence_pause_extra_ms: int = Field(default=200, ge=0, le=1000)
    comma_pause_scale: float = Field(default=0.8, ge=0, le=2)
    min_chars: int = Field(default=24, ge=1, le=1000)
    max_chars: int = Field(default=160, ge=1, le=1000)
    lookahead: int = Field(default=12, ge=0, le=1000)
    prompt: str = Field(default="", max_length=16000)
    messages: list[ChatMessage] | None = Field(default=None, min_length=1, max_length=61)
    backend: BackendConfig | None = None

    @model_validator(mode="after")
    def validate_conversation(self):
        if self.messages is not None:
            if self.source != "llm" or self.prompt:
                raise ValueError("会話履歴は LLM 入力で prompt と併用せず指定してください。")
            if sum(len(message.content) for message in self.messages) > 16000:
                raise ValueError("会話履歴は合計16,000文字までです。")
            turns = self.messages[1:] if self.messages[0].role == "system" else self.messages
            if not turns or turns[-1].role != "user":
                raise ValueError("最後のメッセージは user にしてください。")
            for index, message in enumerate(turns):
                if message.role != ("user" if index % 2 == 0 else "assistant"):
                    raise ValueError("会話履歴の役割の順序が不正です。")
        return self


class TextEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text", "end"]
    text: str = Field(default="", max_length=8192)


class Session:
    def __init__(self, ws, client, settings, config):
        self.ws, self.client, self.settings, self.config = ws, client, settings, config
        self.input = asyncio.Queue(maxsize=32)
        self.segments = asyncio.Queue(maxsize=8)
        self.audio_slots = asyncio.Semaphore(3)
        self.pending_audio = set()
        self.all_played = asyncio.Event()
        self.all_played.set()
        self.send_lock = asyncio.Lock()
        self.started = time.monotonic()
        self.last_activity = self.started
        self.sequence = 0
        self.received = 0
        self.ended = False
        self.id = uuid4().hex
        self.segmenter = Segmenter(config.mode, config.min_chars, config.max_chars, config.lookahead)

    def elapsed(self):
        return round((time.monotonic() - self.started) * 1000, 1)

    async def send(self, kind, **data):
        async with self.send_lock:
            await asyncio.wait_for(self.ws.send_json({"type": kind, **data}), timeout=15)
            self.last_activity = time.monotonic()

    def acknowledge(self, seq):
        if type(seq) is not int or seq not in self.pending_audio:
            raise ValueError("不明または重複した played ID です。")
        self.pending_audio.remove(seq)
        self.last_activity = time.monotonic()
        self.audio_slots.release()
        if not self.pending_audio:
            self.all_played.set()

    def enqueue(self, event):
        if self.ended:
            raise ValueError("入力はすでに終了しています。")
        self.input.put_nowait(None if event.type == "end" else event.text)
        self.last_activity = time.monotonic()
        if event.type == "end":
            self.ended = True

    async def emit(self, segments):
        for segment in segments:
            self.sequence += 1
            event = {"seq": self.sequence, "text": segment.text, "reason": segment.reason,
                     "confirmed_ms": self.elapsed()}
            await self.send("segment", **event)
            if self.config.audio and segment.text.strip():
                await self.segments.put(event)

    async def accept_text(self, text):
        self.received += len(text)
        if self.received > 100000:
            raise ValueError("1 セッションの入力上限（100,000文字）を超えました。")
        await self.send("delta", text=text, received_ms=self.elapsed())
        await self.segment("feed", text)

    async def segment(self, operation, *args):
        method = getattr(self.segmenter, operation)
        # Model loading/inference must not block playback ACKs, cancel, or other sessions.
        result = (await asyncio.to_thread(method, *args) if self.config.mode == "syntax"
                  else method(*args))
        await self.segmentation_notices()
        for decision in self.segmenter.diagnostics:
            await self.send("boundary", **decision)
        self.segmenter.diagnostics.clear()
        await self.emit(result)

    async def segmentation_notices(self):
        for message in self.segmenter.notices:
            await self.send("warning", code="syntax_fallback", message=message)
        self.segmenter.notices.clear()

    async def produce(self):
        if self.config.source == "llm":
            # A separate feeder allows sentence-idle flushing even while LLM stalls.
            async def feed_llm():
                messages = ([message.model_dump() for message in self.config.messages]
                            if self.config.messages is not None else None)
                async for text in llm_stream(self.client, self.settings, self.config.prompt, messages):
                    await self.input.put(text)
                await self.input.put(None)
            async with asyncio.TaskGroup() as group:
                group.create_task(feed_llm())
                group.create_task(self.consume_input())
        else:
            await self.consume_input()
        await self.send("text_done", segments=self.sequence, elapsed_ms=self.elapsed(),
                        segmentation=self.segmenter.metrics())
        await self.segments.put(None)

    async def consume_input(self):
        while True:
            try:
                text = await asyncio.wait_for(self.input.get(), timeout=0.25)
            except TimeoutError:
                await self.segment("idle")
                continue
            if text is None:
                await self.segment("flush")
                return
            await self.accept_text(text)

    async def render(self):
        while True:
            segment = await self.segments.get()
            if segment is None:
                break
            # Bounded lookahead: client must acknowledge playback, not merely receipt.
            await asyncio.wait_for(self.audio_slots.acquire(), timeout=180)
            start = time.monotonic()
            audio = await synthesize(self.client, self.settings, segment["text"],
                                     self.config.speaker, self.config.speed,
                                     pauses=PauseSettings(self.config.sentence_pause_extra_ms,
                                                          self.config.comma_pause_scale)
                                     if self.config.adjust_pauses else None)
            self.pending_audio.add(segment["seq"])
            self.all_played.clear()
            await self.send("audio", seq=segment["seq"], mime="audio/wav",
                            wav_base64=base64.b64encode(audio).decode("ascii"),
                            synthesis_ms=round((time.monotonic() - start) * 1000, 1),
                            ready_ms=self.elapsed())
        await asyncio.wait_for(self.all_played.wait(), timeout=180)

    async def run(self):
        await self.send("ready", mode=self.config.mode, audio=self.config.audio, session_id=self.id,
                        pause_adjustment={"enabled": self.config.adjust_pauses,
                                          "sentence_extra_ms": self.config.sentence_pause_extra_ms,
                                          "comma_scale": self.config.comma_pause_scale})
        await self.segmentation_notices()
        async with asyncio.TaskGroup() as group:
            group.create_task(self.produce())
            group.create_task(self.render())
        await self.send("done", segments=self.sequence, elapsed_ms=self.elapsed(),
                        segmentation=self.segmenter.metrics())


def safe_error(exc):
    if isinstance(exc, BaseExceptionGroup):
        return safe_error(exc.exceptions[0])
    if isinstance(exc, (ProviderError, ValueError, TypeError)):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "処理または再生確認がタイムアウトしました。"
    return "処理に失敗しました。サーバーを確認してください。"


def create_app(settings=None, client=None, stt=None, voice_vad=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=10), trust_env=False, follow_redirects=False)
        app.state.active = 0
        app.state.sessions = {}
        app.state.stt = stt if stt is not None else create_stt()
        app.state.stt_backends = ({app.state.stt.status().get("backend", "default"): app.state.stt} if stt is not None
                                  else {backend: create_stt(backend) for backend in BACKENDS})
        app.state.stt_busy = False
        app.state.voice_vad = voice_vad if voice_vad is not None else ResidentSilero()
        yield
        await app.state.voice_vad.close()
        if client is None:
            await app.state.client.aclose()

    app = FastAPI(title="KoeLoop — VOICEVOX voice conversations", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def browser_headers(request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
                return JSONResponse({"error": "異なる Origin からは送信できません。"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "microphone=(self), camera=()"
        if request.url.path.startswith("/api/stt/"):
            response.headers["Cache-Control"] = "no-store"
        if request.url.path in {"/", "/lab"} or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "chat.html")

    @app.get("/lab")
    async def lab():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    async def config():
        # URL credentials/query strings are never exposed through browser defaults.
        try:
            llm_url = backend_url(settings.llm_base_url, llm=True)
        except ValueError:
            llm_url = ""
        try:
            voicevox_url = backend_url(settings.voicevox_url)
        except ValueError:
            voicevox_url = ""
        return {"llm_model": settings.llm_model, "llm_configured": bool(settings.llm_model),
                "llm_url": llm_url, "voicevox_url": voicevox_url,
                 "llm_reasoning_effort": settings.llm_reasoning_effort,
                "syntax": {"installed": syntax_installed(), "model": MODEL},
                "stt": app.state.stt.status(),
                "voice_vad": app.state.voice_vad.status(),
                "stt_backends": [{"id": backend, **service.status()}
                                 for backend, service in app.state.stt_backends.items()],
                "auth": False, "max_sessions": settings.max_sessions}

    @app.websocket("/ws/vad")
    async def streaming_vad(ws: WebSocket):
        await voice_websocket(ws, app)

    @app.post("/api/stt/transcribe")
    async def transcribe(request: Request):
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "audio/wav":
            return JSONResponse({"error": "Content-Type: audio/wav で送信してください。"}, status_code=415)
        backend = request.headers.get("x-stt-backend")
        service = app.state.stt if backend is None else app.state.stt_backends.get(backend)
        if service is None:
            return JSONResponse({"error": "未対応の STT 選択です。画面を再読み込みしてください。"}, status_code=400)
        stt_status = service.status()
        if not stt_status["available"]:
            return JSONResponse({"error": stt_status.get("message", "STT が未設定です。サーバー設定を確認してください。")}, status_code=503)
        try:
            if int(request.headers.get("content-length", "0")) > MAX_BYTES:
                return JSONResponse({"error": "録音データが上限を超えています。"}, status_code=413)
        except ValueError:
            return JSONResponse({"error": "音声のリクエストサイズが不正です。"}, status_code=400)
        if app.state.stt_busy:
            return JSONResponse({"error": "別の音声を処理中です。少し待って録音し直してください。"},
                                status_code=429, headers={"Retry-After": "1"})
        # Reserve before upload: only one bounded recording or inference at a time.
        app.state.stt_busy = True
        inference = disconnect = None
        try:
            data = bytearray()
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > MAX_BYTES:
                        raise STTError("録音データが上限を超えています。最大30秒で録音してください。", 413)
                    data.extend(chunk)
            recording = await asyncio.to_thread(validate_recording, bytes(data))
            inference = asyncio.create_task(service.transcribe(recording))
            # After consuming the body, wait for the ASGI disconnect directly.
            # Nonblocking is_disconnected() polling can miss it through HTTP middleware.
            async def wait_for_disconnect():
                while True:
                    if (await request.receive())["type"] == "http.disconnect":
                        return
            disconnect = asyncio.create_task(wait_for_disconnect())
            finished, _ = await asyncio.wait({inference, disconnect}, return_when=asyncio.FIRST_COMPLETED)
            if disconnect in finished and inference not in finished:
                return JSONResponse({"error": "文字起こしを中止しました。"}, status_code=499)
            text = await inference
            return {"text": text, "language": "ja", "duration_seconds": recording.seconds}
        except STTError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        except TimeoutError:
            return JSONResponse({"error": "音声のアップロードがタイムアウトしました。"}, status_code=408)
        except ClientDisconnect:
            return JSONResponse({"error": "音声入力の接続が切れました。"}, status_code=499)
        finally:
            for task in (inference, disconnect):
                if task:
                    task.cancel()
            await asyncio.gather(*(task for task in (inference, disconnect) if task), return_exceptions=True)
            app.state.stt_busy = False

    @app.get("/api/speakers")
    async def speakers():
        try:
            response = await app.state.client.get(f"{settings.voicevox_url}/speakers")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return JSONResponse({"error": "VOICEVOX Engine に接続できません。"}, status_code=502)

    @app.post("/api/backends/speakers")
    async def selected_speakers(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 8192:
                return JSONResponse({"error": "リクエストが大きすぎます。"}, status_code=413)
        try:
            selected = SpeakerRequest.model_validate_json(body)
        except ValidationError:
            return JSONResponse({"error": "有効な VOICEVOX の HTTP(S) URL を指定してください。"}, status_code=422)
        try:
            response = await app.state.client.get(f"{selected.voicevox_url}/speakers", timeout=10)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return JSONResponse({"error": "VOICEVOX に接続できません。URL と Engine の起動状態を確認してください。"}, status_code=502)

    @app.post("/api/sessions/{session_id}/events")
    async def external_input(session_id: str, request: Request):
        # A browser explicitly opens an external-input session; POST feeds that browser.
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"error": "異なる Origin からは送信できません。"}, status_code=403)
        session = app.state.sessions.get(session_id)
        if session is None or session.config.source != "external":
            return JSONResponse({"error": "外部入力セッションが見つかりません。"}, status_code=404)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return JSONResponse({"error": "メッセージが大きすぎます。"}, status_code=413)
        try:
            event = TextEvent.model_validate_json(body)
        except ValidationError:
            return JSONResponse({"error": "text（最大8,192文字）または end を送信してください。"}, status_code=422)
        try:
            session.enqueue(event)
        except asyncio.QueueFull:
            return JSONResponse({"error": "入力待ちが満杯です。同じイベントを少し待って再送してください。"},
                                status_code=429, headers={"Retry-After": "1"})
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"accepted": True}, status_code=202)

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        # No auth by request. Still reject foreign browser origins (cross-site WS hijacking).
        origin = ws.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            scheme = "https" if ws.url.scheme == "wss" else "http"
            if parsed.scheme != scheme or parsed.netloc != ws.headers.get("host") or parsed.path:
                await ws.close(code=1008)
                return
        if app.state.active >= settings.max_sessions:
            await ws.close(code=1013)
            return
        app.state.active += 1
        runner = reader = None
        session = None
        try:
            await ws.accept()
            first = await asyncio.wait_for(ws.receive_text(), timeout=15)
            if len(first.encode("utf-8")) > 65536:
                raise ValueError("開始メッセージが大きすぎます。")
            try:
                config = Start.model_validate_json(first)
            except ValidationError as exc:
                raise ValueError(start_error_message(exc)) from None
            if config.source == "llm" and not config.prompt.strip() and config.messages is None:
                raise ValueError("LLM 用の prompt または messages が必要です。")
            selected_settings = config.backend.resolve(settings) if config.backend else settings
            session = Session(ws, app.state.client, selected_settings, config)
            app.state.sessions[session.id] = session
            runner = asyncio.create_task(session.run())
            reader = asyncio.create_task(ws.receive_text())
            while True:
                remaining = max(0.1, 180 - (time.monotonic() - session.last_activity))
                finished, _ = await asyncio.wait({runner, reader}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if not finished:
                    if time.monotonic() - session.last_activity >= 180:
                        raise TimeoutError()
                    continue
                if runner in finished:
                    await runner
                    break
                message = reader.result()
                if len(message) > 65536:
                    raise ValueError("メッセージが大きすぎます。")
                try:
                    event = json.loads(message)
                except ValueError:
                    raise ValueError("有効な JSON を送信してください。") from None
                if not isinstance(event, dict):
                    raise TypeError("JSON オブジェクトを送信してください。")
                kind = event.get("type")
                if kind == "cancel":
                    runner.cancel()
                    await asyncio.gather(runner, return_exceptions=True)
                    await session.send("cancelled")
                    break
                if kind == "played":
                    session.acknowledge(event.get("seq"))
                elif kind in {"text", "end"}:
                    if config.source != "text" or session.ended:
                        raise ValueError("この状態ではテキストを送信できません。")
                    try:
                        parsed_event = TextEvent.model_validate(event)
                    except ValidationError:
                        raise ValueError("text は 8,192 文字以下の文字列にしてください。") from None
                    try:
                        session.enqueue(parsed_event)
                    except asyncio.QueueFull:
                        raise ValueError("入力キューが満杯です。送信速度を下げて新しいセッションで試してください。") from None
                else:
                    raise ValueError("不明なイベントです。")
                reader = asyncio.create_task(ws.receive_text())
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - protocol boundary: redact unexpected failures too.
            # Avoid logging upstream URLs, credentials, prompt text, or audio bodies.
            logger.warning("Session failed (%s)", type(exc).__name__)
            if runner:
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)
            try:
                await asyncio.wait_for(ws.send_json({"type": "error", "message": safe_error(exc)}), timeout=5)
            except (RuntimeError, WebSocketDisconnect, TimeoutError, OSError):
                pass
        finally:
            for task in (runner, reader):
                if task:
                    task.cancel()
            await asyncio.gather(*(t for t in (runner, reader) if t), return_exceptions=True)
            app.state.active -= 1
            if session:
                app.state.sessions.pop(session.id, None)
            try:
                await ws.close()
            except (RuntimeError, WebSocketDisconnect, OSError):
                pass

    return app


app = create_app()
