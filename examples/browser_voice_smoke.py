"""Actual browser mic/worklet/WS -> VAD -> Parakeet -> one LLM turn -> TTS.

Uses a fake microphone and deterministic STT/LLM/Engine. No real cloud calls.
--real-stt uses the installed Silero VAD and Parakeet instead of test doubles.
"""

import argparse
import asyncio
import io
import json
import shutil
import wave
from pathlib import Path

import httpx
from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from examples.stt_live import to_16k
from restream.app import create_app
from restream.providers import Settings
from restream.vad import ResidentSilero


class FakeVoice:
    def __init__(self):
        self.mode = "endpoint"
        self.text = "好きな色は青です。"
        self.count = 0
        self.frames = 0
        self.starts = 0
        self.closed = 0
        self.failed = False
        self.settings = []

    def status(self):
        return {"available": True, "streaming": True}

    async def start(self, settings):
        self.settings.append(settings.model_dump(exclude={"type"}))
        self.count = 0
        self.frames = 0
        self.starts += 1
        return self

    def feed(self, pcm):
        assert 0 < len(pcm) <= 6400 and len(pcm) % 2 == 0
        self.count += 1
        self.frames += len(pcm) // 2
        if self.failed:
            raise RuntimeError("test native failure")
        return self.end() if self.mode == "endpoint" and self.count >= 18 else {
            "type": "state", "state": "speaking" if self.count >= 3 else "waiting"}

    def end(self):
        return {"type": "end", "speech": bool(self.text) and self.frames > 0,
                "start_sample": 0, "end_sample": self.frames if self.text else 0}

    async def close(self):
        self.closed += 1


class FakeRecognizer:
    def __init__(self, voice):
        self.voice = voice

    def status(self):
        return {"backend": "parakeet", "available": True, "engine_label": "Test Parakeet"}

    async def transcribe(self, recording):
        return self.voice.text


async def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    voice = ResidentSilero() if args.real_stt else FakeVoice()
    if not voice.status()["available"]:
        raise RuntimeError(voice.status()["message"])
    audio_fixture = None
    if args.real_stt:
        audio_fixture = output / "fake-microphone.wav"
        audio_fixture.write_bytes(to_16k(Path(args.audio).read_bytes()))
    calls, speech, errors, socket_events, mic_checks = [], [], [], [], []

    def upstream(request):
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            calls.append(body)
            first = json.dumps({"choices": [{"delta": {"content": "青ですね。"}}]}, ensure_ascii=False)
            second = json.dumps({"choices": [{"delta": {"content": "覚えておきます。"}}]}, ensure_ascii=False)
            return httpx.Response(200, text=f"data: {first}\n\ndata: {second}\n\ndata: [DONE]\n\n")
        if request.url.path == "/audio_query":
            speech.append(request.url.params["text"])
            return httpx.Response(200, json={})
        if request.url.path == "/synthesis":
            data = io.BytesIO()
            with wave.open(data, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x01\x00" * 2400)
            return httpx.Response(200, content=data.getvalue())
        raise AssertionError(request.url)

    settings = Settings(llm_base_url="http://llm.example/v1", llm_model="voice-test", voicevox_url="http://engine.example")
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client, serve(
        create_app(settings, client=client, voice_vad=voice, stt=None if args.real_stt else FakeRecognizer(voice))
    ) as base, async_playwright() as playwright:
        flags = ["--use-fake-device-for-media-stream"]
        if audio_fixture:
            flags.append(f"--use-file-for-fake-audio-capture={audio_fixture.resolve()}%noloop")
        browser = await playwright.chromium.launch(executable_path=shutil.which("chromium"), headless=True,
                                                    chromium_sandbox=True, args=flags)
        try:
            context = await browser.new_context(permissions=["microphone"], viewport={"width": 1280, "height": 1000})
            await context.add_init_script("""
              window.__streams = [];
              window.__llmMicStopped = [];
              window.__delay = 0;
              window.__shortLimit = false;
              const timeout = window.setTimeout.bind(window);
              window.setTimeout = (fn, delay, ...args) => timeout(fn, window.__shortLimit && delay === 30000 ? 1200 : delay, ...args);
              const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
              navigator.mediaDevices.getUserMedia = async (options) => {
                const stream = await gum(options);
                window.__streams.push(stream);
                if (window.__delay) await new Promise(r => setTimeout(r, window.__delay));
                return stream;
              };
              const send = WebSocket.prototype.send;
              WebSocket.prototype.send = function(data) {
                if (this.url.endsWith('/ws') && typeof data === 'string') {
                  const e = JSON.parse(data);
                  if (e.type === 'start') window.__llmMicStopped.push(
                    window.__streams.length > 0 && window.__streams.every(s => s.getTracks().every(t => t.readyState === 'ended')));
                }
                return send.call(this, data);
              };
            """)
            page = await context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            def socket_open(ws):
                def record(frame):
                    event = json.loads(frame)
                    if event["type"] != "audio":
                        socket_events.append({"socket": "vad" if ws.url.endswith("/ws/vad") else "llm", **event})
                ws.on("framereceived", record)
            page.on("websocket", socket_open)
            await page.goto(base)
            await expect(page.locator("#voiceStart")).to_be_enabled()
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("発話中", timeout=15000)
            await expect(page.locator("#voiceStatus")).to_contain_text("発話中", timeout=15000)
            await expect(page.locator("#prompt")).to_have_value("")
            await expect(page.locator(".message.user")).to_have_count(0)
            assert not calls, "speech was submitted before recognition"
            await expect(page.locator("#send")).to_be_disabled()
            await expect(page.locator("#llmUrl")).to_be_disabled()
            await page.screenshot(path=str(output / "partial.png"))
            await expect(page.locator(".message.user")).to_have_count(1, timeout=20000)
            await expect(page.locator("#status")).to_have_text("完了", timeout=20000)
            await expect(page.locator("#voiceStart")).to_be_enabled()
            assert len(calls) == 1 and len(speech) == 2
            assert all(await page.evaluate("window.__llmMicStopped"))
            finals = [event for event in socket_events if event["socket"] == "vad" and event["type"] == "end"]
            assert len(finals) == 1
            assert calls[0]["messages"][-1]["content"].strip()
            assert any(event["type"] == "state" and event.get("state") == "speaking" for event in socket_events)
            assert not any("text" in event for event in socket_events if event["socket"] == "vad")
            await page.screenshot(path=str(output / "conversation.png"))
            streams_after_response = await page.evaluate("window.__streams.length")
            await page.wait_for_timeout(300)
            assert await page.evaluate("window.__streams.length") == streams_after_response, "microphone auto-reopened"

            if not args.real_stt:
                # Second turn reuses the conversation, not a fresh prompt-only request.
                voice.text = "好きな色は何ですか。"
                await page.locator("#voiceStart").click()
                await expect(page.locator(".message.user")).to_have_count(2, timeout=15000)
                await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
                assert len(calls) == 2
                assert calls[1]["messages"][1:4] == [
                    {"role": "user", "content": "好きな色は青です。"},
                    {"role": "assistant", "content": "青ですね。覚えておきます。"},
                    {"role": "user", "content": "好きな色は何ですか。"}]
                count = len(calls)
                voice.mode = "manual"
                voice.text = "まだ途中です。"
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("発話中")
                await page.locator("#voiceCancel").click()
                await expect(page.locator("#prompt")).to_have_value("")
                await page.wait_for_timeout(300)
                assert len(calls) == count
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("発話中")
                await page.locator("#newChat").click()
                await expect(page.locator(".message")).to_have_count(0)
                await page.wait_for_timeout(300)
                assert len(calls) == count

                # Empty speech is never a user message.
                voice.text = ""
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("発話中")
                await page.wait_for_timeout(600)
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("発話を検出できません")
                assert len(calls) == count
                # Explicit manual end does submit a non-empty final once.
                voice.text = "手動で確定します。"
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("発話中")
                await page.locator("#voiceStart").click()
                await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
                assert len(calls) == count + 1
                count = len(calls)

                # Do not mix typed drafts into an automatically submitted voice turn.
                await page.locator("#prompt").fill("手入力の下書き")
                starts = voice.starts
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("下書き")
                assert voice.starts == starts and len(calls) == count
                await expect(page.locator("#prompt")).to_have_value("手入力の下書き")
                await page.locator("#prompt").fill("")

                # Permission resolving after cancel cannot create a new STT/LLM session.
                await page.evaluate("window.__delay = 500")
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceCancel")).to_be_visible()
                await expect(page.locator("#voiceStatus")).to_contain_text("使用許可を待っています")
                ready_starts = voice.starts
                await page.locator("#voiceCancel").click()
                await page.wait_for_timeout(700)
                await page.wait_for_function("window.__streams.every(s => s.getTracks().every(t => t.readyState === 'ended'))")
                assert voice.starts == ready_starts and len(calls) == count
                await page.evaluate("window.__delay = 0")
                # Exercise the duration-limit branch without a 30s browser-test sleep.
                # The actual 480000-sample worklet limit is covered by node tests.
                await page.evaluate("window.__shortLimit = true")
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("30秒上限", timeout=15000)
                assert len(calls) == count
                await expect(page.locator("#prompt")).to_have_value("")
                await page.locator("#prompt").fill("")
                await page.evaluate("window.__shortLimit = false")
                voice.failed = True
                await page.locator("#voiceStart").click()
                await expect(page.locator("#voiceStatus")).to_contain_text("失敗", timeout=15000)
                assert len(calls) == count
            mic_checks = await page.evaluate("window.__llmMicStopped")
            assert all(mic_checks) and not errors
            report = {"status": "passed", "real_silero_vad": args.real_stt, "physical_microphone": False,
                      "real_parakeet": args.real_stt, "no_partial_transcripts": True,
                      "sent_text": calls[0]["messages"][-1]["content"],
                      "llm": "deterministic mock", "voicevox": "deterministic mock",
                      "llm_turns": len(calls), "tts_segments": len(speech), "mic_stopped_before_llm": mic_checks,
                      "page_errors": errors, "events": socket_events}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: value for key, value in report.items() if key != "events"}, ensure_ascii=False))
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-stt", action="store_true")
    parser.add_argument("--refine", action="store_true", help=argparse.SUPPRESS)  # Old command-line alias.
    parser.add_argument("--audio", default="artifacts/stt-number-fixtures/walk.wav")
    parser.add_argument("--output", default="artifacts/voice-browser")
    args = parser.parse_args()
    if args.refine and not args.real_stt:
        parser.error("--refine requires --real-stt and a Parakeet-enabled environment")
    asyncio.run(main(args))
