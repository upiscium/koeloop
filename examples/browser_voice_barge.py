"""Bounded browser E2E checks for continuous voice barge-in.

Run with::

    nix develop .#browser -c python -m examples.browser_voice_barge

Chromium uses its fake microphone and the VAD, Parakeet, LLM, and VOICEVOX
providers are deterministic local test doubles. No cloud or physical mic is
used. The test exercises the real browser worklet, WebSockets, PCM TTS playback,
and chat UI.
"""

import asyncio
import io
import json
import shutil
import time
import wave
from pathlib import Path

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from examples.browser_voice_continuous import (
    INSTRUMENTATION,
    assert_no_new_vad_start,
    event_count,
    start_continuous,
    trace,
    wait_for_event_count,
    wait_for_tracks_ended,
)
from restream.app import create_app
from restream.providers import Settings


class ControlledVADTurn:
    """One fake VAD socket whose speaking and endpoint transitions are test-settable."""

    def __init__(self, owner, number, endpoint_after_frames=None):
        self.owner = owner
        self.number = number
        self.endpoint_after_frames = endpoint_after_frames
        self.frames = 0
        self.state = "waiting"
        self.speaking_requested_at = None
        self.speaking_at_frames = None
        self.endpoint_requested_at = None
        self.endpoint_event = None

    def mark_speaking(self):
        if self.speaking_requested_at is None:
            self.speaking_requested_at = time.monotonic()
        self.state = "speaking"

    def request_endpoint(self):
        if self.endpoint_requested_at is None:
            self.endpoint_requested_at = time.monotonic()

    def _end(self):
        self.endpoint_event = {
            "start_sample": 0,
            "end_sample": self.frames,
            "speech": self.frames > 0,
        }
        return {"type": "end", **self.endpoint_event}

    def feed(self, pcm):
        assert 0 < len(pcm) <= 6400 and len(pcm) % 2 == 0
        self.frames += len(pcm) // 2
        if self.speaking_at_frames is None and self.state == "speaking":
            self.speaking_at_frames = self.frames
        if (self.endpoint_requested_at is not None
                or (self.endpoint_after_frames is not None and self.frames >= self.endpoint_after_frames)):
            return self._end()
        return {"type": "state", "state": self.state}

    def end(self):
        if self.state == "speaking":
            return self._end()
        self.endpoint_event = {"start_sample": 0, "end_sample": 0, "speech": False}
        return {"type": "end", **self.endpoint_event}


class ControllableVAD:
    """Creates per-WebSocket fake turns; only explicitly planned turns endpoint."""

    def __init__(self):
        self.plans = []
        self.turns = []
        self.settings = []
        self.starts = 0
        self.closed = 0
        self.hold_turn = None
        self.setup_entered = asyncio.Event()

    def status(self):
        return {"available": True, "streaming": True, "engine": "controllable-test-vad"}

    def endpoint_next_after(self, frames=16000):
        self.plans.append(frames)

    async def start(self, settings):
        endpoint_after = self.plans.pop(0) if self.plans else None
        turn = ControlledVADTurn(self, len(self.turns) + 1, endpoint_after)
        self.turns.append(turn)
        self.starts += 1
        self.settings.append(settings.model_dump(exclude={"type"}))
        if turn.number == self.hold_turn:
            self.setup_entered.set()
            await asyncio.Event().wait()  # Cancellation of this socket retires setup.
        return turn

    async def close(self):
        self.closed += 1


class FakeParakeet:
    """Records the accepted WAV sample range and returns queued final transcripts."""

    def __init__(self):
        self.texts = []
        self.accepted = []

    def status(self):
        return {"backend": "parakeet", "available": True, "engine_label": "Test Parakeet"}

    async def transcribe(self, recording):
        with wave.open(io.BytesIO(recording.wav), "rb") as source:
            frames = source.getnframes()
            assert source.getframerate() == 16000
            assert source.getnchannels() == 1 and source.getsampwidth() == 2
        self.accepted.append({"frames": frames, "seconds": recording.seconds, "rms": recording.rms})
        return self.texts.pop(0) if self.texts else "テスト発話です。"

    def queue_texts(self, *texts):
        self.texts.extend(texts)


def _wav_fixture(duration_seconds=3.0):
    """Produce real mono PCM WAV for an observable three-second Web Audio reply."""
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * round(24000 * duration_seconds))
    return data.getvalue()


EXTRA_INSTRUMENTATION = r"""
(() => {
  window.__gumConstraints = [];
  window.__micSourceStreams = [];
  const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    window.__gumConstraints.push(JSON.parse(JSON.stringify(constraints)));
    return getUserMedia(constraints);
  };
  const createMediaStreamSource = AudioContext.prototype.createMediaStreamSource;
  AudioContext.prototype.createMediaStreamSource = function(stream, ...args) {
    window.__micSourceStreams.push(stream);
    return createMediaStreamSource.call(this, stream, ...args);
  };
})();
"""


async def _wait_until(predicate, timeout=10, message="condition not reached"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(message)
        await asyncio.sleep(0.02)


async def _wait_socket_event(events, source, event_type, count, timeout=10):
    await _wait_until(
        lambda: sum(event["socket"] == source and event["type"] == event_type for event in events) >= count,
        timeout,
        f"timed out waiting for {source} {event_type} #{count}",
    )
    return [event for event in events if event["socket"] == source and event["type"] == event_type][count - 1]


def _socket_event_count(events, source, event_type):
    return sum(event["socket"] == source and event["type"] == event_type for event in events)


def _record_socket_events(page, events, vad_sockets):
    def opened(ws):
        if ws.url.endswith("/ws/vad"):
            source = "vad"
            vad_sockets.append(ws)
        elif ws.url.endswith("/ws"):
            source = "chat"
        else:
            return

        def received(frame):
            if not isinstance(frame, str):
                return
            try:
                message = json.loads(frame)
            except ValueError:
                return
            kind = message.get("type")
            if kind not in {"ready", "state", "end", "text_done", "done", "error", "cancelled"}:
                return
            events.append({
                "socket": source,
                "type": kind,
                "state": message.get("state"),
                "speech": message.get("speech"),
                "start_sample": message.get("start_sample"),
                "end_sample": message.get("end_sample"),
                "t": time.monotonic(),
            })

        ws.on("framereceived", received)

    page.on("websocket", opened)


async def _begin_session(page, voice, *, timeout=15):
    """Start a continuous turn planned to endpoint, and wait for its live monitor."""
    turn_base = len(voice.turns)
    vad_base = await event_count(page, "vad_start")
    stt_base = await event_count(page, "stt_request")
    llm_base = await event_count(page, "llm_start")
    audio_base = await event_count(page, "audio_active")
    done_base = await event_count(page, "llm_done")
    worklet_base = await page.evaluate("window.__micWorklets")
    gum_base = await page.evaluate("window.__streams.length")
    voice.endpoint_next_after(16000)
    await start_continuous(page)
    await wait_for_event_count(page, "stt_request", stt_base + 1, timeout=timeout * 1000)
    await wait_for_event_count(page, "llm_start", llm_base + 1, timeout=timeout * 1000)
    try:
        await wait_for_event_count(page, "vad_start", vad_base + 2, timeout=timeout * 1000)
    except PlaywrightError as exc:
        details = await page.evaluate("""() => ({
          voice: document.querySelector('#voiceStatus').textContent,
          chat: document.querySelector('#status').textContent,
          draft: document.querySelector('#prompt').value,
          cancelHidden: document.querySelector('#voiceCancel').hidden,
          recent: window.__trace.slice(-5)
        })""")
        raise AssertionError(f"monitor/listen did not start: {details}") from exc
    await _wait_until(lambda: len(voice.turns) >= turn_base + 2, timeout,
                      "initial VAD and barge monitor were not created")
    return {
        "turn_base": turn_base,
        "initial_vad_start": vad_base,
        "stt_base": stt_base,
        "llm_base": llm_base,
        "audio_base": audio_base,
        "done_base": done_base,
        "worklet_base": worklet_base,
        "gum_base": gum_base,
        "initial_vad": voice.turns[turn_base],
        "monitor": voice.turns[turn_base + 1],
        "monitor_socket": None,
    }


async def _audio_event(page, session):
    await wait_for_event_count(page, "audio_active", session["audio_base"] + 1, timeout=15000)
    events = await trace(page)
    audio_events = [event for event in events if event["type"] == "audio_active"]
    active = audio_events[session["audio_base"]]
    started = next(event for event in reversed(events) if event["type"] == "audio_start" and event["t"] <= active["t"])
    return {**active, "duration": started["duration"]}


async def _monitor_pcm_during_playback(page, active_audio):
    await page.wait_for_function(
        "(after) => window.__trace.some(event => event.type === 'vad_pcm' && event.t > after)",
        arg=active_audio["t"], timeout=5000,
    )
    events = await trace(page)
    return next(event for event in events
                if event["type"] == "vad_pcm" and event["t"] > active_audio["t"])


async def _mic_state(page):
    return await page.evaluate("""() => ({
      streamCount: window.__streams.length,
      constraints: window.__gumConstraints,
      sourceCount: window.__micSourceStreams.length,
      currentStreamSourceCount: window.__streams.length > 0
        ? window.__micSourceStreams.filter(stream => stream === window.__streams.at(-1)).length : 0,
      allCurrentSourcesUseLatestStream: window.__streams.length > 0 &&
        window.__micSourceStreams.filter(stream => stream === window.__streams.at(-1))
          .every(stream => stream === window.__streams.at(-1)),
      worklets: window.__micWorklets,
      tracks: window.__streams.at(-1)?.getAudioTracks().map(track => ({
        enabled: track.enabled, readyState: track.readyState,
      })) || [],
      audioGraphConnected: window.__trace.at(-1)?.audioGraphConnected || false,
    })""")


def _assert_monitor_capture(mic, pcm_event, gum_base):
    assert mic["streamCount"] == gum_base + 1, f"barge monitoring reacquired the microphone: {mic}"
    assert len(mic["constraints"]) == gum_base + 1
    assert mic["constraints"][-1]["audio"]["echoCancellation"] is True, (
        f"browser AEC was not requested: {mic['constraints']}"
    )
    assert mic["currentStreamSourceCount"] >= 2 and mic["allCurrentSourcesUseLatestStream"], (
        f"initial listen and monitor did not reuse the same capture stream: {mic}"
    )
    assert mic["tracks"] and all(track["enabled"] and track["readyState"] == "live" for track in mic["tracks"]), (
        f"microphone track was not enabled during monitoring: {mic['tracks']}"
    )
    assert pcm_event["audioGraphConnected"], "monitor PCM was sent without the microphone capture graph"
    assert pcm_event["tracks"] and pcm_event["tracks"][-1]["enabled"]
    assert pcm_event["tracks"][-1]["readyState"] == "live", "monitor PCM was sent with an ended track"


def _assert_accepted_range(turn, accepted):
    endpoint = turn.endpoint_event
    assert endpoint is not None, "the controllable VAD did not emit its endpoint"
    assert endpoint["start_sample"] == 0, f"monitor range did not retain onset from sample zero: {endpoint}"
    assert endpoint["end_sample"] == accepted["frames"], (
        f"Parakeet WAV range disagreed with the VAD sample range: endpoint={endpoint}, accepted={accepted}"
    )
    assert turn.speaking_at_frames is not None and turn.speaking_at_frames > 0
    assert accepted["frames"] > turn.speaking_at_frames, (
        "accepted Parakeet audio did not include samples preceding the controlled speaking detection"
    )
    assert accepted["seconds"] == accepted["frames"] / 16000


async def _stop_and_check(page, voice, vad_baseline, *, new_chat=False):
    if new_chat:
        await page.locator("#newChat").click()
        await expect(page.locator(".message")).to_have_count(0)
    else:
        await page.locator("#voiceCancel").click()
        await expect(page.locator("#voiceCancel")).to_be_hidden(timeout=5000)
    await wait_for_tracks_ended(page)
    await assert_no_new_vad_start(page, 700)
    # A monitor may already have been scheduled between the baseline snapshot
    # and Playwright delivering the click. The negative client-side assertion
    # above forbids any new start after cancellation; allow that one in-flight setup.
    assert vad_baseline <= voice.starts <= vad_baseline + 1, (
        f"cancel/new-chat left extra VAD rearms: baseline {vad_baseline}, got {voice.starts}"
    )


async def main(output_path="artifacts/voice-barge-browser"):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    voice = ControllableVAD()
    parakeet = FakeParakeet()
    llm_calls, synthesis_texts, page_errors = [], [], []
    socket_events, vad_sockets = [], []
    failure_gate = asyncio.Event()
    report = {
        "status": "running",
        "physical_microphone": False,
        "external_services": False,
        "checks": [],
        "page_errors": page_errors,
    }
    last_trace = []
    browser = page = None

    async def upstream(request):
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            llm_calls.append(body)
            if body["model"] == "fail-model":
                # Fail only after the test has observed the monitor's ready handshake.
                await failure_gate.wait()
                return httpx.Response(503)
            prompt = body["messages"][-1]["content"]
            if prompt == "割り込み準備中に完了する発話です。":
                await asyncio.wait_for(voice.setup_entered.wait(), 10)
            replies = {
                "中断される最初の発話です。": "中断前に生成された全文です。",
                "割り込み後の質問です。": "割り込み後の返答です。",
                "自然完了の最初の発話です。": "自然完了した返答です。",
                "自然完了後の発話です。": "次の発話を受け付けました。",
                "キャンセルされる発話です。": "キャンセル対象の返答です。",
                "新しい会話で破棄する発話です。": "新しい会話で破棄する返答です。",
                "非表示で破棄する発話です。": "非表示で破棄する返答です。",
                "失敗する発話です。": "この返答は生成されません。",
                "無音モニターを待つ発話です。": "無音モニター中の返答です。",
                "状態なしで確定する発話です。": "状態イベントなしの返答です。",
            }
            text = replies.get(prompt, "テスト用の返答です。")
            event = {"choices": [{"delta": {"content": text}}]}
            return httpx.Response(200, text="data: " + json.dumps(event, ensure_ascii=False) + "\n\ndata: [DONE]\n\n")
        if request.url.path == "/audio_query":
            synthesis_texts.append(request.url.params.get("text", ""))
            return httpx.Response(200, json={})
        if request.url.path == "/synthesis":
            return httpx.Response(200, content=_wav_fixture(), headers={"content-type": "audio/wav"})
        raise AssertionError(f"unexpected deterministic upstream request: {request.url}")

    settings = Settings(
        llm_base_url="http://llm.example/v1",
        llm_model="voice-barge-browser-test",
        voicevox_url="http://engine.example",
        llm_reasoning_effort="none",
    )
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client, serve(
            create_app(settings, client=client, stt=parakeet, voice_vad=voice)
        ) as base, async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=shutil.which("chromium"),
                headless=True,
                chromium_sandbox=True,
                args=["--use-fake-device-for-media-stream"],
            )
            context = await browser.new_context(
                permissions=["microphone"], viewport={"width": 1280, "height": 1000}
            )
            await context.add_init_script(INSTRUMENTATION + "\n" + EXTRA_INSTRUMENTATION)
            page = await context.new_page()
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            _record_socket_events(page, socket_events, vad_sockets)
            await page.goto(base)
            await expect(page.locator("#voiceContinuous")).to_be_enabled(timeout=15000)

            # Default off remains opt-in and loading/reloading never opens a mic.
            await expect(page.locator("#voiceBargeIn")).not_to_be_checked()
            assert voice.starts == 0 and await page.evaluate("window.__streams.length") == 0
            await assert_no_new_vad_start(page, 700)
            await page.locator("#voiceBargeIn").check()
            await page.reload()
            await expect(page.locator("#voiceBargeIn")).to_be_checked(timeout=15000)
            assert voice.starts == 0 and await page.evaluate("window.__streams.length") == 0
            report["checks"].append("barge-in defaults off, does not auto-start, and the enabled preference persists")

            # With barge disabled, a text-only continuous reply has no concurrent monitor.
            await page.locator("#voiceBargeIn").uncheck()
            await page.locator("#audioEnabled").uncheck()
            parakeet.queue_texts("割り込み無効の発話です。")
            vad_base = voice.starts
            vad_start_base = await event_count(page, "vad_start")
            stt_base = await event_count(page, "stt_request")
            llm_base = await event_count(page, "llm_start")
            done_base = await event_count(page, "llm_done")
            voice.endpoint_next_after(16000)
            await start_continuous(page)
            await wait_for_event_count(page, "stt_request", stt_base + 1)
            await wait_for_event_count(page, "llm_start", llm_base + 1)
            await wait_for_event_count(page, "llm_done", done_base + 1)
            await page.wait_for_function("""(count) => {
              const done = window.__trace.filter(event => event.type === 'llm_done');
              if (done.length < count) return false;
              const age = performance.now() - done[count - 1].t;
              return age >= 100 && age < 350 &&
                !window.__trace.some(event => event.type === 'vad_start' && event.t > done[count - 1].t);
            }""", arg=done_base + 1, timeout=5000)
            assert await event_count(page, "vad_start") == vad_start_base + 1
            assert voice.starts == vad_base + 1
            await page.locator("#voiceCancel").click()
            await wait_for_tracks_ended(page)
            assert len(llm_calls) == 1 and len(parakeet.accepted) == 1
            report["checks"].append("default-off continuous reply does not arm a concurrent VAD monitor")
            await page.locator("#newChat").click()

            # The main barge case: text_done has arrived, PCM is genuinely playing,
            # then the test-controlled monitor state cancels audio immediately but
            # leaves that same VAD stream/worklet/capture alive through endpoint.
            await page.locator("#audioEnabled").check()
            await page.locator("#voiceBargeIn").check()
            await expect(page.locator("#voiceBargeIn")).to_be_checked()
            parakeet.queue_texts("中断される最初の発話です。", "割り込み後の質問です。")
            main_calls_base = len(llm_calls)
            main_accepted_base = len(parakeet.accepted)
            text_done_base = _socket_event_count(socket_events, "chat", "text_done")
            cancel_base = await event_count(page, "llm_cancel")
            session = await _begin_session(page, voice)
            assert voice.settings[session["turn_base"] + 1]["threshold"] >= 0.65
            await _wait_socket_event(socket_events, "chat", "text_done", text_done_base + 1)
            active = await _audio_event(page, session)
            await expect(page.locator(".message.assistant .body").last).to_have_text("中断前に生成された全文です。")
            pcm = await _monitor_pcm_during_playback(page, active)
            mic_before_interrupt = await _mic_state(page)
            _assert_monitor_capture(mic_before_interrupt, pcm, session["gum_base"])
            assert active["duration"] >= 2.9, f"TTS fixture was not the required three seconds: {active}"
            assert mic_before_interrupt["worklets"] == session["worklet_base"] + 2, (
                "monitor did not arm a fresh worklet alongside the initial VAD worklet"
            )
            session["monitor_socket"] = vad_sockets[session["turn_base"] + 1]
            assert not session["monitor_socket"].is_closed(), "monitor VAD socket closed before barge-in"
            worklets_before_interrupt = mic_before_interrupt["worklets"]
            sources_before_interrupt = mic_before_interrupt["sourceCount"]
            await _wait_until(lambda: session["monitor"].frames >= 8000, timeout=5,
                              message="monitor did not retain enough pre-detection PCM for an unpadded Parakeet range")

            speaking_at = time.monotonic()
            speaking_perf = await page.evaluate("performance.now()")
            session["monitor"].mark_speaking()
            await expect(page.locator("#voiceStatus")).to_contain_text("割り込み", timeout=5000)
            await wait_for_event_count(page, "llm_cancel", cancel_base + 1, timeout=5000)
            cancel_trace = await trace(page)
            cancel_event = [event for event in cancel_trace if event["type"] == "llm_cancel"][cancel_base]
            assert time.monotonic() - speaking_at < 1.0, "speaking monitor did not cancel the old reply immediately"
            await page.wait_for_function(
                "(start) => window.__trace.some(e => e.type === 'audio_end' && e.t > start)",
                arg=active["t"], timeout=3000,
            )
            cancel_trace = await trace(page)
            ended_audio = next(event for event in cancel_trace
                               if event["type"] == "audio_end" and event["t"] > active["t"])
            assert ended_audio["t"] - active["t"] < 2800, (
                f"interrupted PCM was allowed to play naturally: start={active}, end={ended_audio}"
            )
            assert ended_audio["t"] - speaking_perf < 1000, (
                f"old PCM did not stop promptly after monitor speaking: end={ended_audio}"
            )
            assert not session["monitor_socket"].is_closed(), "barge-in replaced/closed the monitor VAD socket"
            mic_after_interrupt = await _mic_state(page)
            assert mic_after_interrupt["streamCount"] == session["gum_base"] + 1
            assert mic_after_interrupt["sourceCount"] == sources_before_interrupt
            assert mic_after_interrupt["worklets"] == worklets_before_interrupt
            assert mic_after_interrupt["allCurrentSourcesUseLatestStream"]
            assert mic_after_interrupt["tracks"] and all(track["enabled"] for track in mic_after_interrupt["tracks"])
            assert mic_after_interrupt["audioGraphConnected"], "capture graph reset at the interruption"
            await page.wait_for_timeout(200)
            no_stale_ack = [event for event in await trace(page)
                            if event["type"] == "played_ack" and event["t"] > cancel_event["t"]]
            assert not no_stale_ack, f"cancelled playback sent a stale played ACK: {no_stale_ack}"

            session["monitor"].request_endpoint()
            await wait_for_event_count(page, "stt_request", session["stt_base"] + 2, timeout=15000)
            await _wait_until(lambda: len(parakeet.accepted) >= main_accepted_base + 2, timeout=15,
                              message="barge follow-up was not recognized exactly once")
            _assert_accepted_range(session["monitor"], parakeet.accepted[main_accepted_base + 1])
            await wait_for_event_count(page, "llm_start", session["llm_base"] + 2, timeout=15000)
            assert len(llm_calls) == main_calls_base + 2 and len(parakeet.accepted) == main_accepted_base + 2
            history = llm_calls[main_calls_base + 1]["messages"]
            assert history[-1] == {"role": "user", "content": "割り込み後の質問です。"}
            assert not any("中断される最初の発話です。" in message["content"]
                           or "中断前に生成された全文です。" in message["content"] for message in history), (
                f"interrupted turn leaked into the next LLM history: {history}"
            )
            await expect(page.locator(".message-note").first).to_contain_text("中断")
            await _stop_and_check(page, voice, voice.starts)
            report["checks"].append(
                "speaking monitor interrupts three-second Web Audio TTS, retains the live monitor/capture, "
                "recognizes one onset-preserving follow-up, and excludes the interrupted completed-text turn from history"
            )
            await page.locator("#newChat").click()

            # Natural success hands the existing monitor directly into ordinary listen:
            # no new WS/worklet after playback done, and a later utterance uses that stream.
            parakeet.queue_texts("自然完了の最初の発話です。", "自然完了後の発話です。")
            natural_calls_base = len(llm_calls)
            natural_accepted_base = len(parakeet.accepted)
            session = await _begin_session(page, voice)
            session["monitor_socket"] = vad_sockets[session["turn_base"] + 1]
            active = await _audio_event(page, session)
            await _monitor_pcm_during_playback(page, active)
            worklets_at_monitor = await page.evaluate("window.__micWorklets")
            source_count_at_monitor = (await _mic_state(page))["sourceCount"]
            await wait_for_event_count(page, "llm_done", session["done_base"] + 1, timeout=15000)
            done_event = next(event for event in await trace(page)
                              if event["type"] == "llm_done" and event["t"] > active["t"])
            assert done_event["t"] - active["t"] >= 2500, "natural done preceded the three-second PCM playback"
            await assert_no_new_vad_start(page, 700)
            assert not session["monitor_socket"].is_closed(), "natural done closed the monitor VAD socket"
            assert await page.evaluate("window.__micWorklets") == worklets_at_monitor
            mic_after_done = await _mic_state(page)
            assert mic_after_done["streamCount"] == session["gum_base"] + 1
            assert mic_after_done["sourceCount"] == source_count_at_monitor
            assert mic_after_done["allCurrentSourcesUseLatestStream"]
            monitor_pcm = await _monitor_pcm_during_playback(page, active)
            _assert_monitor_capture(mic_after_done, monitor_pcm, session["gum_base"])

            session["monitor"].mark_speaking()
            await _wait_socket_event(socket_events, "vad", "state",
                                     _socket_event_count(socket_events, "vad", "state") + 1)
            session["monitor"].request_endpoint()
            await wait_for_event_count(page, "stt_request", session["stt_base"] + 2, timeout=15000)
            await _wait_until(lambda: len(parakeet.accepted) >= natural_accepted_base + 2, timeout=15,
                              message="post-playback utterance was not recognized")
            _assert_accepted_range(session["monitor"], parakeet.accepted[natural_accepted_base + 1])
            await wait_for_event_count(page, "llm_start", session["llm_base"] + 2, timeout=15000)
            assert len(llm_calls) == natural_calls_base + 2
            assert len(parakeet.accepted) == natural_accepted_base + 2
            natural_history = llm_calls[natural_calls_base + 1]["messages"]
            assert {"role": "user", "content": "自然完了の最初の発話です。"} in natural_history
            assert {"role": "assistant", "content": "自然完了した返答です。"} in natural_history
            assert natural_history[-1] == {"role": "user", "content": "自然完了後の発話です。"}
            await _stop_and_check(page, voice, voice.starts)
            report["checks"].append(
                "natural playback completion adopts the same monitor WS/worklet; later speech retains its onset and completed history"
            )
            await page.locator("#newChat").click()

            # If reply completion beats monitor readiness, there is no captured
            # onset to preserve. Retire setup and return via the normal cooldown
            # with the user's original (non-strict) VAD threshold.
            await page.locator("#audioEnabled").uncheck()
            parakeet.queue_texts("割り込み準備中に完了する発話です。")
            voice.hold_turn = len(voice.turns) + 2
            voice.setup_entered.clear()
            starts_before = await event_count(page, "vad_start")
            done_before = await event_count(page, "llm_done")
            settings_before = len(voice.settings)
            voice.endpoint_next_after(16000)
            await start_continuous(page)
            await asyncio.wait_for(voice.setup_entered.wait(), 10)
            await wait_for_event_count(page, "llm_done", done_before + 1)
            await wait_for_event_count(page, "vad_start", starts_before + 3)
            await _wait_until(lambda: len(voice.settings) >= settings_before + 3)
            setup_trace = await trace(page)
            done = [event for event in setup_trace if event["type"] == "llm_done"][done_before]
            fallback = [event for event in setup_trace if event["type"] == "vad_start"][starts_before + 2]
            assert fallback["t"] - done["t"] >= 500
            assert voice.settings[settings_before + 1]["threshold"] >= 0.65
            assert voice.settings[settings_before + 2]["threshold"] == 0.5
            await _stop_and_check(page, voice, voice.starts)
            voice.hold_turn = None
            report["checks"].append("done during pending monitor setup retires it and resumes after cooldown with normal threshold")
            await page.locator("#newChat").click()
            await page.locator("#audioEnabled").check()

            # Explicit cancel and New Chat both invalidate monitor callbacks during real playback.
            for action, prompt in (
                ("cancel", "キャンセルされる発話です。"),
                ("new_chat", "新しい会話で破棄する発話です。"),
            ):
                parakeet.queue_texts(prompt)
                session = await _begin_session(page, voice)
                await _audio_event(page, session)
                vad_baseline = voice.starts
                await _stop_and_check(page, voice, vad_baseline, new_chat=action == "new_chat")
                report["checks"].append(f"{action} during playback closes monitor/mic and never rearms")
                await page.locator("#newChat").click()

            # A hidden page must also release the monitor and never start again on return.
            await page.locator("#audioEnabled").uncheck()
            parakeet.queue_texts("非表示で破棄する発話です。")
            session = await _begin_session(page, voice)
            vad_baseline = voice.starts
            await page.evaluate("""() => {
              Object.defineProperty(document, 'hidden', {configurable: true, value: true});
              document.dispatchEvent(new Event('visibilitychange'));
            }""")
            await wait_for_tracks_ended(page)
            await page.evaluate("""() => {
              Object.defineProperty(document, 'hidden', {configurable: true, value: false});
              document.dispatchEvent(new Event('visibilitychange'));
            }""")
            await assert_no_new_vad_start(page, 700)
            assert voice.starts == vad_baseline
            report["checks"].append("hidden page closes the barge monitor/mic and foreground return does not rearm")
            await page.locator("#newChat").click()

            # Upstream failure after the monitor is armed must close, not resurrect, capture.
            await page.locator("#model").fill("fail-model")
            await page.locator("#model").press("Tab")
            parakeet.queue_texts("失敗する発話です。")
            error_base = _socket_event_count(socket_events, "chat", "error")
            session = await _begin_session(page, voice)
            vad_baseline = voice.starts
            failure_gate.set()
            await _wait_socket_event(socket_events, "chat", "error", error_base + 1, timeout=10)
            await expect(page.locator("#voiceCancel")).to_be_hidden(timeout=10000)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page, 700)
            assert voice.starts == vad_baseline
            await page.locator("#model").fill("voice-barge-browser-test")
            await page.locator("#model").press("Tab")
            report["checks"].append("LLM failure after monitor setup closes capture and rejects stale callbacks")
            await page.locator("#newChat").click()

            # Shorten only the monitor's 30s silence timer. It may recycle, but must
            # not recognize/send silence or terminate the already-playing old reply.
            await page.locator("#audioEnabled").check()
            parakeet.queue_texts("無音モニターを待つ発話です。", "状態なしで確定する発話です。")
            recycle_calls_base = len(llm_calls)
            recycle_accepted_base = len(parakeet.accepted)
            turn_base = len(voice.turns)
            vad_start_base = await event_count(page, "vad_start")
            stt_base = await event_count(page, "stt_request")
            llm_base = await event_count(page, "llm_start")
            audio_base = await event_count(page, "audio_active")
            done_base = await event_count(page, "llm_done")
            gum_base = await page.evaluate("window.__streams.length")
            voice.endpoint_next_after(16000)
            await start_continuous(page)
            await wait_for_event_count(page, "stt_request", stt_base + 1)
            # The initial recording's real 30s limit is already cleared; later monitor
            # timers observe this flag and are reduced to 1.2s by the reused harness.
            await page.evaluate("window.__shortLimit = true")
            await wait_for_event_count(page, "llm_start", llm_base + 1)
            await wait_for_event_count(page, "vad_start", vad_start_base + 2)
            await _wait_until(lambda: len(voice.turns) >= turn_base + 2, 15)
            recycle_session = {
                "stt_base": stt_base,
                "llm_base": llm_base,
                "audio_base": audio_base,
                "turn_base": turn_base,
                "gum_base": gum_base,
                "monitor": voice.turns[turn_base + 1],
            }
            active = await _audio_event(page, recycle_session)
            await wait_for_event_count(page, "vad_start", vad_start_base + 3, timeout=10000)
            await _wait_until(lambda: len(voice.turns) >= turn_base + 3, 10,
                              "silent monitor did not start its replacement VAD turn")
            assert len(parakeet.accepted) == recycle_accepted_base + 1, (
                "silent monitor recycle spuriously invoked Parakeet"
            )
            assert len(llm_calls) == recycle_calls_base + 1, "silent monitor recycle sent an extra LLM turn"
            assert await event_count(page, "llm_done") == done_base, (
                "silent monitor recycle ended the original reply before its three-second playback"
            )
            assert await page.evaluate("window.__streams.length") == gum_base + 1
            await page.evaluate("window.__shortLimit = false")
            recycled_monitor = voice.turns[-1]
            await _wait_until(lambda: recycled_monitor.frames >= 8000, timeout=5,
                              message="recycled silent monitor did not resume PCM capture")
            cancel_base = await event_count(page, "llm_cancel")
            recycled_monitor.request_endpoint()  # Direct end without a preceding speaking state.
            await wait_for_event_count(page, "llm_cancel", cancel_base + 1)
            await wait_for_event_count(page, "stt_request", stt_base + 2, timeout=15000)
            await _wait_until(lambda: len(parakeet.accepted) == recycle_accepted_base + 2, timeout=15,
                              message="direct endpoint after silent recycle was not transcribed once")
            assert recycled_monitor.speaking_at_frames is None, "direct-end case unexpectedly emitted speaking state"
            assert recycled_monitor.endpoint_event["start_sample"] == 0
            direct_recording = parakeet.accepted[recycle_accepted_base + 1]
            assert direct_recording["frames"] == recycled_monitor.endpoint_event["end_sample"]
            await wait_for_event_count(page, "llm_start", llm_base + 2, timeout=15000)
            assert len(llm_calls) == recycle_calls_base + 2
            await page.locator("#voiceCancel").click()
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page, 700)
            report["checks"].append(
                "silent 30s monitor rotation does not ASR/send/end the old reply; direct speech endpoint interrupts without a speaking event"
            )

            assert not page_errors, page_errors
            last_trace = await trace(page)
            report["status"] = "passed"
            await browser.close()
            browser = None
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if page is not None:
            try:
                current_trace = await trace(page)
                if current_trace:
                    last_trace = current_trace
            except PlaywrightError:
                pass
        if browser is not None:
            await browser.close()
        report["page_errors"] = page_errors
        report["counts"] = {
            "llm_turns": len(llm_calls),
            "parakeet_requests": len(parakeet.accepted),
            "accepted_recordings": parakeet.accepted,
            "tts_segments": len(synthesis_texts),
            "vad_sessions": voice.starts,
            "vad_closed": voice.closed,
            "monitor_endpoints": [turn.endpoint_event for turn in voice.turns if turn.endpoint_event],
            "microphone_acquisitions": sum(event["type"] == "gum_acquired" for event in last_trace),
            "worklets_created": sum(event["type"] == "worklet_created" for event in last_trace),
        }
        report["events"] = [
            {key: event.get(key) for key in ("type", "t", "duration", "seq", "phase") if key in event}
            for event in last_trace
            if event["type"] in {
                "continuous_click", "gum_acquired", "mic_source_created", "worklet_created",
                "vad_start", "vad_end", "vad_cancel", "stt_request", "llm_start", "llm_cancel",
                "text_done", "audio_start", "audio_active", "audio_end", "played_ack", "llm_done",
            }
        ]
        report["socket_events"] = [
            event for event in socket_events
            if event["type"] in {"ready", "state", "end", "text_done", "done", "error", "cancelled"}
        ]
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {key: value for key, value in report.items() if key not in {"events", "socket_events"}}
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
