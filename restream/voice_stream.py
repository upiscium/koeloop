"""Single-turn VAD WebSocket. No transcription and no LLM calls."""

import asyncio
import json
from urllib.parse import urlsplit

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .stt import SAMPLE_RATE, STTError
from .vad import VADStart, native_call


async def voice_websocket(ws: WebSocket, app):
    origin = ws.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme != ("https" if ws.url.scheme == "wss" else "http") or parsed.netloc != ws.headers.get("host") or parsed.path:
            await ws.close(code=1008)
            return
    # Share the admission slot with batch STT. Never create an unbounded number of models.
    if app.state.stt_busy:
        await ws.close(code=1013)
        return
    admitted = False
    service = app.state.voice_vad
    queue = asyncio.Queue(maxsize=20)  # <= 4 seconds for max-sized chunks; UI uses 100ms chunks.
    reader = processor = None
    try:
        await ws.accept()
        initial = await asyncio.wait_for(ws.receive_text(), 10)
        if len(initial) > 512:
            raise STTError("VAD設定が大きすぎます。")
        try:
            settings = VADStart.model_validate_json(initial)
        except ValidationError:
            raise STTError("VAD設定が不正です。無音待ち500～3000ms、前後余白0～500ms、しきい値0.1～0.9を指定してください。") from None
        # An idle or invalid handshake must not reserve (or release) STT's slot.
        # There is no await between this check and acquisition.
        if app.state.stt_busy:
            await ws.close(code=1013)
            return
        app.state.stt_busy = True
        admitted = True

        async def read_input():
            total = 0
            ended = False
            while True:
                message = await asyncio.wait_for(ws.receive(), 45 if total == 0 else 15)
                if message["type"] == "websocket.disconnect":
                    return "disconnect"
                if message.get("bytes") is not None:
                    pcm = message["bytes"]
                    if ended or not 0 < len(pcm) <= 6400 or len(pcm) % 2:
                        raise STTError("音声は16kHz PCM16・最大200msのチャンクで送信してください。")
                    total += len(pcm)
                    if total > SAMPLE_RATE * 30 * 2:
                        raise STTError("逐次録音は30秒までです。")
                    value = pcm
                else:
                    text = message.get("text", "")
                    if len(text) > 128:
                        raise STTError("音声制御メッセージが大きすぎます。")
                    event = json.loads(text)
                    if event == {"type": "cancel"}:
                        return "cancel"
                    if event != {"type": "end"} or ended:
                        raise STTError("音声制御は end / cancel のみ利用できます。")
                    value = None
                    ended = True
                try:
                    queue.put_nowait(value)
                except asyncio.QueueFull:
                    raise STTError("逐次認識が入力に追いつきません。録音を中止しました。") from None

        async def process_audio():
            turn = await service.start(settings)
            await ws.send_json({"type": "ready", "sample_rate": SAMPLE_RATE, "max_seconds": 30,
                                "settings": settings.model_dump(exclude={"type"})})
            sequence = 0
            processed_samples = 0
            last = None
            while True:
                pcm = await queue.get()
                event = await native_call(turn.end if pcm is None else turn.feed, *([] if pcm is None else [pcm]))
                if pcm is not None:
                    processed_samples += len(pcm) // 2
                if event.get("type") == "end":
                    start, end = event.get("start_sample"), event.get("end_sample")
                    if (type(event.get("speech")) is not bool or type(start) is not int or type(end) is not int
                            or not 0 <= start <= end <= processed_samples):
                        raise STTError("VADの音声範囲が不正です。")
                    # Construct metadata explicitly: never expose a transcript on this endpoint.
                    await ws.send_json({"type": "end", "speech": event["speech"], "start_sample": start,
                                        "end_sample": end, "reason": "manual" if pcm is None else "endpoint"})
                    return
                if pcm is None or event.get("type") != "state" or event.get("state") not in {"waiting", "speaking"}:
                    raise STTError("VADの状態を確認できませんでした。")
                sequence += 1
                if event["state"] != last:
                    await ws.send_json({"type": "state", "state": event["state"]})
                    last = event["state"]
                await ws.send_json({"type": "ack", "seq": sequence})

        reader = asyncio.create_task(read_input())
        processor = asyncio.create_task(process_audio())
        async with asyncio.timeout(90):
            finished, _ = await asyncio.wait({reader, processor}, return_when=asyncio.FIRST_COMPLETED)
            # User cancel/disconnect wins if it races completion.
            if reader in finished:
                reason = await reader
                if reason == "cancel":
                    await ws.send_json({"type": "cancelled"})
            else:
                await processor
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - protocol boundary, never echo native exceptions or transcripts.
        try:
            await asyncio.wait_for(ws.send_json({"type": "error", "message": str(exc) if isinstance(exc, STTError) else
                                                "発話検出に失敗しました。接続とVADの設定を確認してください。"}), 3)
        except (RuntimeError, OSError, TimeoutError, WebSocketDisconnect):
            pass
    finally:
        for task in (reader, processor):
            if task:
                task.cancel()
        await asyncio.gather(*(task for task in (reader, processor) if task), return_exceptions=True)
        # native_call drains cancelled CPU work before this lease is released.
        # Per-turn state is discarded; no global model reset is needed.
        if admitted:
            app.state.stt_busy = False
        try:
            await ws.close()
        except (RuntimeError, OSError, WebSocketDisconnect):
            pass
