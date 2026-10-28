"""Local Silero ONNX VAD and one-turn endpointing; never transcribes audio.

Each turn owns its recurrent state and framing buffer. The CPU ONNX graph is
shared, so cancellation of one turn cannot contaminate the next turn's state.
"""

import asyncio
import math
import os
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .stt import SAMPLE_RATE, STTError

WINDOW = 512  # Silero's 16kHz input window: 32ms.


async def native_call(function, *args):
    """Keep the caller's admission lease until its native thread has finished.

    Cancellation stops result delivery, not an already-running CPU call. Shield
    the worker and drain it even on repeated cancellation before releasing the
    shared STT slot. The WS reader can still acknowledge cancellation promptly.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - discard native errors only after cancellation.
                break  # Retrieve the exception below; never publish a cancelled result.
        if not worker.cancelled():
            worker.exception()
        raise


class VADStart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["start"]
    silence_ms: int = Field(default=1000, ge=500, le=3000)
    pre_ms: int = Field(default=200, ge=0, le=500)
    post_ms: int = Field(default=200, ge=0, le=500)
    threshold: float = Field(default=0.5, ge=0.1, le=0.9)


class Endpoint:
    """Pure sample-based endpoint policy over a per-window speech scorer."""

    def __init__(self, scorer, settings: VADStart):
        self.scorer, self.settings = scorer, settings
        self.pending = bytearray()
        self.received = self.processed = 0
        self.candidate = None
        self.speech_samples = 0
        self.confirmed = False
        self.silence_start = None
        self.finished = False

    def window(self, pcm, valid_frames):
        probability = float(self.scorer(pcm))
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("invalid VAD score")
        start = self.processed
        self.processed += valid_frames
        if probability >= self.settings.threshold:
            if self.candidate is None:
                self.candidate = start
            self.speech_samples += valid_frames
            self.silence_start = None
            # Reject isolated clicks/noise. Recognition may still be imperfect.
            if self.speech_samples >= SAMPLE_RATE // 4:  # 250ms of positive frames.
                self.confirmed = True
        elif probability < max(0.01, self.settings.threshold - 0.15):
            if not self.confirmed:
                self.candidate = None
                self.speech_samples = 0
            else:
                if self.silence_start is None:
                    self.silence_start = start
                if self.processed - self.silence_start >= self.settings.silence_ms * 16:
                    return self.finish(self.silence_start + self.settings.post_ms * 16)
        return None

    def finish(self, end):
        self.finished = True
        if not self.confirmed:
            return {"type": "end", "speech": False, "start_sample": 0, "end_sample": 0}
        return {"type": "end", "speech": True,
                "start_sample": max(0, self.candidate - self.settings.pre_ms * 16),
                "end_sample": min(self.received, end)}

    def feed(self, pcm):
        if self.finished or not 0 < len(pcm) <= 6400 or len(pcm) % 2:
            raise ValueError("invalid PCM/state")
        self.received += len(pcm) // 2
        if self.received > SAMPLE_RATE * 30:
            raise ValueError("recording limit exceeded")
        self.pending.extend(pcm)
        while len(self.pending) >= WINDOW * 2:
            frame = bytes(self.pending[:WINDOW * 2])
            del self.pending[:WINDOW * 2]
            result = self.window(frame, WINDOW)
            if result:
                return result
        return {"type": "state", "state": "speaking" if self.confirmed else "waiting"}

    def end(self):
        if self.finished:
            raise ValueError("turn already ended")
        if self.pending:
            valid = len(self.pending) // 2
            result = self.window(bytes(self.pending).ljust(WINDOW * 2, b"\x00"), valid)
            self.pending.clear()
            if result:
                return result
        return self.finish(self.received)


class SileroScorer:
    def __init__(self, session):
        import numpy as np

        self.session = session
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, pcm):
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(1, WINDOW) / 32768
        audio = np.concatenate((self.context, audio), axis=1)
        output, state = self.session.run(None, {"input": audio, "state": self.state,
                                                "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self.state = state
        self.context = audio[:, -64:].copy()
        return float(output.reshape(-1)[0])


class ResidentSilero:
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path is not None else os.getenv("STT_VAD_MODEL_PATH", "")
        self.session = None
        self.lock = Lock()
        self.closed = False

    def status(self):
        available = bool(self.model_path and Path(self.model_path).is_file()
                         and find_spec("onnxruntime") and find_spec("numpy"))
        return {"available": available, "engine": "silero-vad", "version": "6.2", "sample_rate": SAMPLE_RATE,
                "message": "Silero VADで発話を検出します。" if available else
                "音声会話にはSilero VADとParakeetが必要です。nix develop .#voice で起動してください。"}

    def make_turn(self, settings):
        if not self.status()["available"]:
            raise STTError(self.status()["message"], 503)
        import onnxruntime as ort

        with self.lock:
            if self.closed:
                raise STTError("VADサーバーは終了しています。", 503)
            if self.session is None:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.log_severity_level = 3
                ort.disable_telemetry_events()
                self.session = ort.InferenceSession(self.model_path, sess_options=options,
                                                    providers=["CPUExecutionProvider"])
            session = self.session
        return Endpoint(SileroScorer(session), settings)

    async def start(self, settings):
        # Loading/inference cannot block WS cancellation or audio playback ACKs.
        return await native_call(self.make_turn, settings)

    async def close(self):
        def shutdown():
            with self.lock:
                self.closed = True
                self.session = None
        await asyncio.to_thread(shutdown)
