"""Local, bounded dictation processes. No LLM calls, saved recordings, or temp files."""

import asyncio
import io
import math
import os
import shutil
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000
MAX_SECONDS = 30
MAX_BYTES = SAMPLE_RATE * MAX_SECONDS * 2 + 8192
MAX_TEXT = 4000
BACKENDS = ("whisper", "parakeet", "whisper-ja", "wav2vec-ja", "vosk")


class STTError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Recording:
    wav: bytes
    seconds: float
    rms: float


def validate_recording(data: bytes) -> Recording:
    if len(data) > MAX_BYTES:
        raise STTError("音声データが上限を超えています。最大30秒で録音してください。", 413)
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            if (source.getnchannels() != 1 or source.getsampwidth() != 2
                    or source.getframerate() != SAMPLE_RATE or source.getcomptype() != "NONE"):
                raise STTError("音声は16kHz・モノラル・16bit PCM WAVで送信してください。")
            count = source.getnframes()
            if not SAMPLE_RATE // 2 <= count <= SAMPLE_RATE * MAX_SECONDS:
                raise STTError("録音時間は0.5～30秒にしてください。")
            pcm = source.readframes(count)
            if len(pcm) != count * 2:
                raise STTError("音声データが途中で切れています。録音し直してください。")
    except (wave.Error, EOFError, ValueError) as exc:
        raise STTError("有効な PCM WAV を送信してください。") from exc
    # Rebuild a minimal WAV; arbitrary uploaded ancillary chunks never reach the decoder.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm)
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    rms = math.sqrt(sum(sample * sample for sample in samples) / count) / 32768
    return Recording(buffer.getvalue(), count / SAMPLE_RATE, rms)


@dataclass(frozen=True)
class STTConfig:
    command: str = "whisper-cli"
    model_path: str = ""
    threads: int = 4
    timeout: float = 60

    @classmethod
    def from_env(cls):
        return cls(command=os.getenv("STT_COMMAND", "whisper-cli"),
                   model_path=os.getenv("STT_MODEL_PATH", ""),
                   threads=max(1, min(8, int(os.getenv("STT_THREADS", "4")))))


class ProcessSTT:
    """Shared subprocess supervision: per-recording memory, cancellation and limits."""

    config: STTConfig

    def status(self):
        raise NotImplementedError

    def command_args(self):
        raise NotImplementedError

    def process_environment(self):
        return None

    async def transcribe(self, recording: Recording) -> str:
        if not self.status()["available"]:
            raise STTError("ローカル STT が未設定です。サーバーの STT 設定を確認してください。", 503)
        # Avoid common silence hallucinations. This is only an energy gate, not a speech VAD.
        if recording.rms < 0.001:
            return ""
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            *self.command_args(), env=self.process_environment(),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ))
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            # A cancellation during spawn must not lose the child's handle.
            try:
                process = await spawn
            except OSError:
                pass
            else:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()
            raise
        except OSError as exc:
            raise STTError("STT プロセスを起動できません。サーバー設定を確認してください。", 503) from exc

        async def write_audio():
            try:
                process.stdin.write(recording.wav)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Report the process exit code below, without leaking stderr.
            finally:
                process.stdin.close()

        async def read_text():
            result = bytearray()
            while chunk := await process.stdout.read(4096):
                result.extend(chunk)
                if len(result) > 65536:
                    raise STTError("認識結果が長すぎます。短く録音し直してください。", 422)
            return bytes(result)

        tasks = [asyncio.create_task(write_audio()), asyncio.create_task(read_text()),
                 asyncio.create_task(process.wait())]
        try:
            _, output, code = await asyncio.wait_for(asyncio.gather(*tasks), self.config.timeout)
            if code != 0:
                raise STTError("音声認識に失敗しました。選択した STT のモデルと実行環境を確認してください。", 502)
            try:
                text = output.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise STTError("STT が有効なテキストを返しませんでした。", 502) from exc
            if len(text) > MAX_TEXT:
                raise STTError("認識結果が4000文字を超えました。短く録音し直してください。", 422)
            return text
        except TimeoutError as exc:
            raise STTError("音声認識がタイムアウトしました。短く録音し直してください。", 504) from exc
        finally:
            # Cancellation/disconnect also kills and reaps our own subprocess.
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await process.wait()


class WhisperSTT(ProcessSTT):
    def __init__(self, config=None):
        self.config = config or STTConfig.from_env()

    def status(self):
        available = bool(shutil.which(self.config.command) and self.config.model_path
                         and Path(self.config.model_path).is_file())
        return {"available": available, "engine": "whisper.cpp", "language": "ja",
                "engine_label": "Whisper / whisper.cpp・CPU", "backend": "whisper",
                "max_seconds": MAX_SECONDS, "sample_rate": SAMPLE_RATE,
                "message": "ローカルSTT・日本語。録音停止後に文字起こしします。自動送信しません。" if available else
                "STT が未設定です。nix develop でサーバーを起動するか STT_MODEL_PATH と STT_COMMAND を設定してください。"}

    def command_args(self):
        return (self.config.command, "--model", self.config.model_path,
                "--file", "-", "--language", "ja", "--threads", str(self.config.threads),
                "--processors", "1", "--no-gpu", "--no-prints", "--no-timestamps",
                "--output-txt", "--output-file", "-",
                "--suppress-nst", "--best-of", "5", "--beam-size", "5", "--no-fallback")


class UnavailableSTT:
    def status(self):
        return {"available": False, "message": "STT_BACKEND は " + " / ".join(BACKENDS) + " を指定してください。"}

    async def transcribe(self, recording):
        raise STTError(self.status()["message"], 503)


def create_stt(backend=None):
    backend = backend if backend is not None else os.getenv("STT_BACKEND", "whisper")
    if backend == "whisper":
        return WhisperSTT()
    if backend == "parakeet":
        from .parakeet import ParakeetSTT

        return ParakeetSTT()
    if backend in {"whisper-ja", "wav2vec-ja", "vosk"}:
        from .japanese_stt import JapaneseSTT

        return JapaneseSTT(backend)
    # Do not silently use another model if a selected backend is misspelled.
    return UnavailableSTT()
