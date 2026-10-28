"""Bounded browser E2E checks for continuous voice turns.

Run in the browser development shell with::

    nix develop .#browser -c python -m examples.browser_voice_continuous

Chromium uses its fake microphone. VAD, Parakeet, the LLM, and VOICEVOX are
deterministic local test doubles; no cloud, real Engine, or external service is
used.
"""

import asyncio
import io
import json
import shutil
import wave
from pathlib import Path

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from examples.browser_voice_smoke import FakeVoice
from restream.app import create_app
from restream.providers import Settings
from restream.stt import STTError


class FakeParakeet:
    """A controllable Parakeet stand-in for successful and failed ASR turns."""

    def __init__(self):
        self.text = "好きな色は青です。"
        self.failed = False
        self.delay = 0
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.calls = []

    def status(self):
        return {"backend": "parakeet", "available": True, "engine_label": "Test Parakeet"}

    async def transcribe(self, recording):
        self.calls.append(recording.seconds)
        self.entered.set()
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.failed:
            raise STTError("deterministic Parakeet failure", 502)
        return self.text


def _wav_fixture(duration_seconds=1.2):
    """Return a real, decodable PCM WAV long enough to observe playback gating."""
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * round(24000 * duration_seconds))
    return data.getvalue()


INSTRUMENTATION = r"""
(() => {
  window.__trace = [];
  window.__streams = [];
  window.__micWorklets = 0;
  window.__gumDelay = 0;
  window.__shortLimit = false;
  const nativeTimeout = window.setTimeout.bind(window);
  window.setTimeout = (callback, delay, ...args) => nativeTimeout(
    callback, window.__shortLimit && delay === 30000 ? 1200 : delay, ...args);
  window.__continuousRequested = false;
  window.__phase = "idle";

  const edges = new Map();
  const nodeIds = new WeakMap();
  const nodeTags = new WeakMap();
  let nextNodeId = 1;
  const nodeId = (node) => {
    if (!nodeIds.has(node)) nodeIds.set(node, nextNodeId++);
    return nodeIds.get(node);
  };
  const state = () => {
    const tracks = window.__streams.flatMap((stream) => stream.getAudioTracks()).map((track) => ({
      enabled: track.enabled, readyState: track.readyState,
    }));
    const cancel = document.querySelector("#voiceCancel");
    const settings = document.querySelector("#settingsFields");
    const send = document.querySelector("#send");
    const mic = document.querySelector("#mic");
    const draft = document.querySelector("#prompt");
    return {
      streamCount: window.__streams.length,
      tracks,
      audioGraphConnected: edges.size > 0,
      cancelVisible: !!cancel && !cancel.hidden,
      cancelText: cancel?.textContent.trim() || "",
      settingsDisabled: !!settings?.disabled,
      sendDisabled: !!send?.disabled,
      micDisabled: !!mic?.disabled,
      draftBlocked: !!draft && (draft.disabled || draft.readOnly),
    };
  };
  const record = (type, extra = {}) => window.__trace.push({
    type,
    t: performance.now(),
    phase: window.__phase,
    ...state(),
    ...extra,
  });

  if (window.AudioNode) {
    const connect = AudioNode.prototype.connect;
    const disconnect = AudioNode.prototype.disconnect;
    AudioNode.prototype.connect = function(destination, ...args) {
      const result = connect.call(this, destination, ...args);
      if (nodeTags.has(this)) {
        edges.set(`${nodeId(this)}>${nodeId(destination)}`, {from: this, to: destination});
      }
      return result;
    };
    AudioNode.prototype.disconnect = function(...args) {
      const result = disconnect.call(this, ...args);
      if (nodeTags.has(this)) {
        for (const [key, edge] of edges) {
          if (edge.from === this && (!args.length || edge.to === args[0])) edges.delete(key);
        }
      }
      return result;
    };
  }

  if (window.AudioContext) {
    const createMicSource = AudioContext.prototype.createMediaStreamSource;
    AudioContext.prototype.createMediaStreamSource = function(...args) {
      const source = createMicSource.apply(this, args);
      nodeTags.set(source, "microphone");
      record("mic_source_created");
      return source;
    };
    const createBufferSource = AudioContext.prototype.createBufferSource;
    const nativeBufferStart = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function(...startArgs) {
      const source = this;
      const startAt = typeof startArgs[0] === "number" ? startArgs[0] : source.context.currentTime;
      window.__phase = "tts";
      record("audio_start", {duration: source.buffer?.duration || 0});
      const notePlaybackStart = () => {
        if (source.context.state !== "running") return requestAnimationFrame(notePlaybackStart);
        if (source.context.currentTime >= startAt) return record("audio_active");
        requestAnimationFrame(notePlaybackStart);
      };
      requestAnimationFrame(notePlaybackStart);
      return nativeBufferStart.apply(source, startArgs);
    };
    AudioContext.prototype.createBufferSource = function(...args) {
      const source = createBufferSource.apply(this, args);
      source.addEventListener("ended", () => record("audio_end"), {once: true});
      return source;
    };
  }

  if (window.AudioWorkletNode) {
    const NativeAudioWorkletNode = window.AudioWorkletNode;
    window.AudioWorkletNode = new Proxy(NativeAudioWorkletNode, {
      construct(target, args) {
        const node = Reflect.construct(target, args, target);
        if (args[1] === "voice-pcm") {
          nodeTags.set(node, "voice-worklet");
          window.__micWorklets++;
          record("worklet_created");
        }
        return node;
      },
    });
  }

  const nativeGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    const stream = await nativeGetUserMedia(constraints);
    window.__streams.push(stream);
    record("gum_acquired");
    if (window.__gumDelay) await new Promise((resolve) => setTimeout(resolve, window.__gumDelay));
    return stream;
  };

  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const resource = args[0];
    const url = typeof resource === "string" ? resource : resource?.url || "";
    if (new URL(url, location.href).pathname === "/api/stt/transcribe") {
      window.__phase = "asr";
      record("stt_request");
    }
    return nativeFetch(...args);
  };

  const NativeWebSocket = window.WebSocket;
  const nativeSend = NativeWebSocket.prototype.send;
  NativeWebSocket.prototype.send = function(data) {
    const isVad = this.url.endsWith("/ws/vad");
    const isChat = this.url.endsWith("/ws");
    let event = null;
    if (typeof data === "string") {
      try { event = JSON.parse(data); } catch (_) { /* Non-protocol text. */ }
    }
    if (isVad) {
      if (event?.type === "start") {
        window.__phase = "listening";
        record("vad_start");
      } else if (event?.type === "end") {
        record("vad_end");
      } else if (event?.type === "cancel") {
        window.__phase = "idle";
        record("vad_cancel");
      } else if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
        record("vad_pcm", {bytes: data.byteLength || data.length || 0});
      }
    } else if (isChat) {
      if (event?.type === "start") {
        window.__phase = "llm";
        record("llm_start");
      } else if (event?.type === "played") {
        record("played_ack", {seq: event.seq});
      } else if (event?.type === "cancel") {
        record("llm_cancel");
      }
    }
    return nativeSend.call(this, data);
  };

  const messageDescriptor = Object.getOwnPropertyDescriptor(NativeWebSocket.prototype, "onmessage");
  if (messageDescriptor?.set && messageDescriptor?.get) {
    Object.defineProperty(NativeWebSocket.prototype, "onmessage", {
      configurable: true,
      enumerable: messageDescriptor.enumerable,
      get() { return messageDescriptor.get.call(this); },
      set(handler) {
        if (typeof handler !== "function") return messageDescriptor.set.call(this, handler);
        const socket = this;
        return messageDescriptor.set.call(this, function(message) {
          if (typeof message.data === "string") {
            try {
              const event = JSON.parse(message.data);
              if (socket.url.endsWith("/ws")) {
                if (event.type === "audio") {
                  window.__phase = "tts";
                  record("llm_audio", {seq: event.seq});
                } else if (event.type === "done") {
                  window.__phase = "cooldown";
                  record("llm_done");
                } else if (event.type === "error" || event.type === "cancelled") {
                  window.__phase = "failed";
                  record(`llm_${event.type}`);
                }
              }
            } catch (_) { /* The app handles malformed frames. */ }
          }
          return handler.call(socket, message);
        });
      },
    });
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest("#voiceContinuous")) {
      window.__continuousRequested = true;
      record("continuous_click");
    } else if (event.target.closest("#voiceCancel")) {
      record("cancel_click");
      window.__continuousRequested = false;
    } else if (event.target.closest("#newChat")) {
      record("new_chat_click");
      window.__continuousRequested = false;
    }
  });

  setInterval(() => {
    if (window.__continuousRequested && ["asr", "llm", "tts", "cooldown"].includes(window.__phase)) {
      record("phase_sample");
    }
  }, 40);
})();
"""


async def wait_for_event_count(page, event_type, count, timeout=15000):
    await page.wait_for_function(
        "({type, count}) => window.__trace.filter((event) => event.type === type).length >= count",
        arg={"type": event_type, "count": count},
        timeout=timeout,
    )


async def trace(page):
    return await page.evaluate("() => window.__trace")


async def event_count(page, event_type):
    return await page.evaluate(
        "(type) => window.__trace.filter((event) => event.type === type).length", event_type
    )


async def start_continuous(page):
    await page.locator("#voiceContinuous").click()
    await expect(page.locator("#voiceCancel")).to_be_visible(timeout=5000)
    await expect(page.locator("#voiceCancel")).to_contain_text("会話を終了")


async def assert_busy_controls(page):
    await expect(page.locator("#send")).to_be_disabled()
    await expect(page.locator("#llmUrl")).to_be_disabled()
    await expect(page.locator("#mic")).to_be_disabled()
    assert await page.locator("#prompt").evaluate("(draft) => draft.disabled || draft.readOnly")


async def wait_for_tracks_ended(page, timeout=5000):
    await page.wait_for_function(
        "() => window.__streams.length > 0 && window.__streams.every((stream) => "
        "stream.getTracks().every((track) => track.readyState === 'ended'))",
        timeout=timeout,
    )


async def assert_no_new_vad_start(page, duration_ms=700):
    await page.evaluate(
        "(duration) => { window.__quietWindow = {baseline: window.__trace.filter(e => e.type === 'vad_start').length, "
        "until: performance.now() + duration}; }",
        duration_ms,
    )
    # This is a bounded negative assertion: poll until either an unwanted start
    # appears or the observation window has elapsed, then inspect which occurred.
    await page.wait_for_function(
        "() => { const q = window.__quietWindow; const starts = window.__trace.filter(e => e.type === 'vad_start').length; "
        "return starts !== q.baseline || performance.now() >= q.until; }",
        timeout=duration_ms + 2000,
    )
    state = await page.evaluate(
        "() => ({baseline: window.__quietWindow.baseline, "
        "starts: window.__trace.filter(e => e.type === 'vad_start').length})"
    )
    assert state["starts"] == state["baseline"], f"unexpected automatic VAD restart: {state}"


async def stop_in_cooldown(page, done_count):
    await page.wait_for_function(
        "(count) => { const done = window.__trace.filter(e => e.type === 'llm_done'); "
        "if (done.length < count) return false; const last = done[count - 1]; "
        "const age = performance.now() - last.t; "
        "return age >= 100 && age < 350 && !window.__trace.some(e => e.type === 'vad_start' && e.t > last.t); }",
        arg=done_count,
        timeout=5000,
    )
    await expect(page.locator("#voiceCancel")).to_be_visible()
    await expect(page.locator("#voiceCancel")).to_contain_text("会話を終了")
    await page.locator("#voiceCancel").click()
    cancel_time = await page.evaluate(
        "() => window.__trace.filter(e => e.type === 'cancel_click').at(-1)?.t"
    )
    done_time = await page.evaluate(
        "(count) => window.__trace.filter(e => e.type === 'llm_done')[count - 1]?.t", done_count
    )
    assert cancel_time is not None and done_time is not None
    assert cancel_time - done_time < 500, "cancel was not exercised inside the 500ms echo cooldown"
    await expect(page.locator("#voiceCancel")).to_be_hidden()


def assert_muted_during_work(trace_events):
    relevant = []
    for event in trace_events:
        phase = event.get("phase")
        if phase not in {"asr", "llm", "tts", "cooldown"}:
            continue
        if event["type"] in {
            "stt_request", "llm_start", "llm_audio", "audio_start", "audio_active", "audio_end", "played_ack",
            "llm_done", "phase_sample",
        }:
            relevant.append(event)
    assert any(event["type"] == "stt_request" for event in relevant), "ASR checkpoint was not observed"
    assert any(event["type"] == "llm_start" for event in relevant), "LLM checkpoint was not observed"
    assert any(event["type"] == "audio_active" for event in relevant), "actual Web Audio playback was not observed"
    assert any(event["type"] == "llm_done" for event in relevant), "server done checkpoint was not observed"
    for event in relevant:
        tracks = event["tracks"]
        assert tracks and all(not track["enabled"] and track["readyState"] == "live" for track in tracks), (
            f"microphone track was not disabled/live during {event['type']} ({event['phase']}): {event}"
        )
        assert not event["audioGraphConnected"], (
            f"microphone audio graph remained connected during {event['type']} ({event['phase']})"
        )
        assert event["cancelVisible"] and "会話を終了" in event["cancelText"], (
            f"continuous cancel control was absent during {event['type']} ({event['phase']})"
        )
        assert event["settingsDisabled"] and event["sendDisabled"] and event["micDisabled"] and event["draftBlocked"], (
            f"busy controls were not blocked during {event['type']} ({event['phase']})"
        )
    assert not any(
        event["type"] == "vad_pcm" and event.get("phase") in {"asr", "llm", "tts", "cooldown"}
        for event in trace_events
    ), "VAD PCM was sent during ASR, LLM, TTS, or echo cooldown"


def assert_playback_gated_restarts(trace_events, completed_turns):
    done_events = [event for event in trace_events if event["type"] == "llm_done"]
    vad_starts = [event for event in trace_events if event["type"] == "vad_start"]
    assert len(done_events) >= completed_turns
    for done in done_events[:completed_turns]:
        next_start = next((event for event in vad_starts if event["t"] > done["t"]), None)
        assert next_start is not None, "successful done did not lead to the next continuous listen"
        assert next_start["t"] - done["t"] >= 500, "continuous listening restarted before the 500ms cooldown"
        preceding_start = next(
            event for event in reversed(trace_events)
            if event["type"] == "llm_start" and event["t"] < done["t"]
        )
        assert any(
            event["type"] == "played_ack" and preceding_start["t"] < event["t"] < done["t"]
            for event in trace_events
        ), "server done arrived before a real playback ACK"
        assert any(
            event["type"] == "audio_end" and preceding_start["t"] < event["t"] < done["t"]
            for event in trace_events
        ), "server done arrived before actual PCM playback completed"
        assert not any(
            event["type"] == "vad_pcm" and done["t"] < event["t"] < next_start["t"]
            for event in trace_events
        ), "VAD PCM was sent during the post-playback echo cooldown"


async def main(output_path="artifacts/voice-continuous-browser"):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    voice = FakeVoice()
    parakeet = FakeParakeet()
    llm_calls, synthesis_queries, synthesis_texts, page_errors = [], [], [], []
    report = {
        "status": "running",
        "physical_microphone": False,
        "external_services": False,
        "checks": [],
        "counts": {},
        "page_errors": page_errors,
    }
    last_trace = []

    def upstream(request):
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            llm_calls.append(body)
            if body["model"] == "fail-model":
                return httpx.Response(503)
            turn = len(llm_calls)
            # Change the recognized text before the next automatic VAD listen.
            if turn == 1:
                voice.text = parakeet.text = "次は天気について聞きます。"
            elif turn == 2:
                voice.text = parakeet.text = "三回目は送信しません。"
            answer = json.dumps(
                {"choices": [{"delta": {"content": f"応答{turn}です。"}}]}, ensure_ascii=False
            )
            return httpx.Response(200, text=f"data: {answer}\n\ndata: [DONE]\n\n")
        if request.url.path == "/audio_query":
            synthesis_texts.append(request.url.params.get("text", ""))
            return httpx.Response(200, json={})
        if request.url.path == "/synthesis":
            synthesis_queries.append(json.loads(request.content))
            return httpx.Response(200, content=_wav_fixture(), headers={"content-type": "audio/wav"})
        raise AssertionError(f"unexpected deterministic upstream request: {request.url}")

    settings = Settings(
        llm_base_url="http://llm.example/v1",
        llm_model="continuous-browser-test",
        voicevox_url="http://engine.example",
        llm_reasoning_effort="none",
    )
    browser = page = None
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
            await context.add_init_script(INSTRUMENTATION)
            page = await context.new_page()
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            await page.goto(base)
            await expect(page.locator("#voiceContinuous")).to_be_enabled(timeout=15000)
            await expect(page.locator("#voiceStart")).to_be_enabled()
            await page.reload()
            await expect(page.locator("#voiceContinuous")).to_be_enabled(timeout=15000)
            assert voice.starts == 0 and await page.evaluate("window.__streams.length") == 0
            await assert_no_new_vad_start(page, 700)
            assert voice.starts == 0
            report["checks"].append("no continuous capture or VAD auto-start on initial reload")

            # Two successful automatic turns prove history retention and that a
            # single microphone stream is reused while the VAD socket/worklet is fresh.
            voice.mode = "endpoint"
            voice.text = parakeet.text = "好きな色は青です。"
            before_done = await event_count(page, "llm_done")
            before_starts = await event_count(page, "vad_start")
            before_worklets = await event_count(page, "worklet_created")
            before_gum = await event_count(page, "gum_acquired")
            await start_continuous(page)
            await assert_busy_controls(page)
            await wait_for_event_count(page, "llm_done", before_done + 2, timeout=35000)
            await expect(page.locator(".message.user")).to_have_count(2, timeout=5000)
            assert len(llm_calls) == 2, f"expected two automatic LLM turns, got {len(llm_calls)}"
            assert llm_calls[0]["messages"][-1]["content"] == "好きな色は青です。"
            assert llm_calls[1]["messages"][1:4] == [
                {"role": "user", "content": "好きな色は青です。"},
                {"role": "assistant", "content": "応答1です。"},
                {"role": "user", "content": "次は天気について聞きます。"},
            ], "second automatic turn did not include completed conversation history"
            await wait_for_event_count(page, "vad_start", before_starts + 3, timeout=10000)
            await wait_for_event_count(page, "worklet_created", before_worklets + 3, timeout=5000)
            before_main_pcm = await event_count(page, "vad_pcm")
            await wait_for_event_count(page, "vad_pcm", before_main_pcm + 1, timeout=5000)
            await expect(page.locator("#voiceCancel")).to_contain_text("会話を終了")
            main_session_trace = await trace(page)
            assert sum(event["type"] == "gum_acquired" for event in main_session_trace) - before_gum == 1, (
                "continuous turns acquired more than one microphone stream"
            )
            assert sum(event["type"] == "worklet_created" for event in main_session_trace) - before_worklets == 3, (
                "each continuous listen must create a fresh VAD worklet"
            )
            assert_muted_during_work(main_session_trace)
            assert_playback_gated_restarts(main_session_trace, 2)
            report["checks"].extend([
                "two automatic turns retain completed user/assistant history",
                "one microphone acquisition reused across three listens; fresh VAD worklet per listen",
                "tracks disabled and microphone graph disconnected during ASR/LLM/TTS/cooldown",
                "no VAD PCM during busy phases; next listen follows playback ACK and 500ms cooldown",
                "busy session blocks settings, manual send, normal mic, and draft edits",
            ])

            # Explicit stop while the third listen is active must end the same
            # stream without submitting another turn or automatically restarting.
            llm_count = len(llm_calls)
            await page.locator("#voiceCancel").click()
            await expect(page.locator("#voiceCancel")).to_be_hidden()
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert len(llm_calls) == llm_count
            report["checks"].append("explicit stop during listening ends the stream without ASR/LLM/restart")

            # Stop during actual Web Audio playback. The long PCM fixture means
            # this is inside a real playback interval, not just a received frame.
            await page.locator("#newChat").click()
            voice.text = parakeet.text = "返答中に終了します。"
            voice.mode = "endpoint"
            starts_before = await event_count(page, "vad_start")
            gum_before = await event_count(page, "gum_acquired")
            done_before = await event_count(page, "llm_done")
            audio_before = await event_count(page, "audio_active")
            await start_continuous(page)
            await wait_for_event_count(page, "audio_active", audio_before + 1, timeout=15000)
            await expect(page.locator("#voiceCancel")).to_contain_text("会話を終了")
            active_audio = await page.evaluate(
                "() => window.__trace.filter(e => e.type === 'audio_start').at(-1)"
            )
            assert active_audio["duration"] >= 1.0, f"PCM fixture too short to verify reply stop: {active_audio}"
            await page.locator("#voiceCancel").click()
            await expect(page.locator("#voiceCancel")).to_be_hidden()
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert await event_count(page, "vad_start") == starts_before + 1
            assert await event_count(page, "gum_acquired") == gum_before + 1
            assert len(llm_calls) == 3 and await event_count(page, "llm_done") == done_before
            assert await event_count(page, "llm_cancel") >= 1, "stop during reply did not cancel its chat socket"
            report["checks"].append("explicit stop during actual TTS playback ends reply and prevents restart")

            # Stop during the 500ms post-playback echo cooldown. The event gate
            # waits for the real server done frame after played ACKs.
            await page.locator("#newChat").click()
            voice.text = parakeet.text = "再生完了後に終了します。"
            starts_before = await event_count(page, "vad_start")
            done_before = await event_count(page, "llm_done")
            await start_continuous(page)
            await wait_for_event_count(page, "llm_done", done_before + 1, timeout=20000)
            await stop_in_cooldown(page, done_before + 1)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert await event_count(page, "vad_start") == starts_before + 1
            report["checks"].append("explicit stop during echo cooldown prevents the next listen")

            # A final Parakeet failure must not retry recognition, send to the
            # LLM, or start another listen.
            await page.locator("#newChat").click()
            voice.text = parakeet.text = "認識に失敗する発話です。"
            voice.mode = "endpoint"
            parakeet.failed = True
            starts_before = await event_count(page, "vad_start")
            stt_before = await event_count(page, "stt_request")
            llm_before = len(llm_calls)
            parakeet_calls_before = len(parakeet.calls)
            await start_continuous(page)
            await wait_for_event_count(page, "stt_request", stt_before + 1, timeout=15000)
            await expect(page.locator("#voiceCancel")).to_be_hidden(timeout=15000)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page, 800)
            assert await event_count(page, "stt_request") == stt_before + 1
            assert len(parakeet.calls) == parakeet_calls_before + 1, "failed ASR was retried"
            assert len(llm_calls) == llm_before, "failed ASR was submitted to the LLM"
            assert await event_count(page, "vad_start") == starts_before + 1
            parakeet.failed = False
            report["checks"].append("ASR error makes one recognition request and does not auto-retry")

            # The legacy voiceStart control can explicitly end a continuous
            # listen; that successful turn still waits through playback and cooldown.
            await page.locator("#newChat").click()
            voice.mode = "manual"
            voice.text = parakeet.text = "手動で発話を確定します。"
            starts_before = await event_count(page, "vad_start")
            pcm_before = await event_count(page, "vad_pcm")
            done_before = await event_count(page, "llm_done")
            llm_before = len(llm_calls)
            await start_continuous(page)
            await wait_for_event_count(page, "vad_pcm", pcm_before + 1, timeout=10000)
            await expect(page.locator("#voiceStart")).to_be_enabled()
            await page.locator("#voiceStart").click()
            await wait_for_event_count(page, "llm_done", done_before + 1, timeout=20000)
            assert len(llm_calls) == llm_before + 1
            assert llm_calls[-1]["messages"][-1]["content"] == "手動で発話を確定します。"
            await stop_in_cooldown(page, done_before + 1)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert await event_count(page, "vad_start") == starts_before + 1
            report["checks"].append("legacy voiceStart manually ends a continuous listen and submits once")

            # New-chat while listening invalidates the session, clears history,
            # and cannot leave a stale completion callback that restarts capture.
            await page.locator("#newChat").click()
            voice.mode = "manual"
            voice.text = parakeet.text = "新しい会話で破棄します。"
            starts_before = await event_count(page, "vad_start")
            pcm_before = await event_count(page, "vad_pcm")
            llm_before = len(llm_calls)
            stt_before = len(parakeet.calls)
            await start_continuous(page)
            await wait_for_event_count(page, "vad_pcm", pcm_before + 1, timeout=10000)
            await page.locator("#newChat").click()
            await expect(page.locator(".message")).to_have_count(0)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert len(llm_calls) == llm_before and len(parakeet.calls) == stt_before
            assert await event_count(page, "vad_start") == starts_before + 1
            report["checks"].append("new chat during listening ends capture and rejects stale auto-restart")

            # Delay a real fake-device permission result after acquisition, then
            # cancel while the app still awaits it. The late stream must be closed.
            await page.evaluate("window.__gumDelay = 1200")
            starts_before = await event_count(page, "vad_start")
            gum_before = await event_count(page, "gum_acquired")
            await start_continuous(page)
            await wait_for_event_count(page, "gum_acquired", gum_before + 1, timeout=10000)
            await page.locator("#voiceCancel").click()
            await expect(page.locator("#voiceCancel")).to_be_hidden()
            await wait_for_tracks_ended(page, timeout=5000)
            await assert_no_new_vad_start(page)
            assert await event_count(page, "vad_start") == starts_before + 1
            await page.evaluate("window.__gumDelay = 0")
            report["checks"].append("late microphone permission after cancel is closed without a stale restart")

            # Healthy silent windows rotate without recognizing or sending;
            # the same safety limit with detected speech must instead end the session.
            voice.mode = "manual"
            voice.text = ""
            await page.evaluate("window.__shortLimit = true")
            starts_before = await event_count(page, "vad_start")
            gum_before = await event_count(page, "gum_acquired")
            stt_before, llm_before = len(parakeet.calls), len(llm_calls)
            await start_continuous(page)
            await wait_for_event_count(page, "vad_start", starts_before + 2)
            assert len(parakeet.calls) == stt_before and len(llm_calls) == llm_before
            assert await event_count(page, "gum_acquired") == gum_before + 1
            await page.locator("#voiceCancel").click()
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            report["checks"].append("silent windows recycle without ASR/LLM or reacquiring the microphone")

            voice.text = "speech at duration limit"
            starts_before = await event_count(page, "vad_start")
            await start_continuous(page)
            await expect(page.locator("#voiceStatus")).to_contain_text("30秒上限", timeout=10000)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert len(parakeet.calls) == stt_before and len(llm_calls) == llm_before
            assert await event_count(page, "vad_start") == starts_before + 1
            await page.evaluate("window.__shortLimit = false")
            report["checks"].append("speech at the duration limit ends continuous mode without ASR/LLM")

            # Hiding and returning to the page must never implicitly rearm capture.
            pcm_before = await event_count(page, "vad_pcm")
            await start_continuous(page)
            await wait_for_event_count(page, "vad_pcm", pcm_before + 1)
            await page.evaluate("""() => {
              Object.defineProperty(document, 'hidden', {configurable: true, value: true});
              document.dispatchEvent(new Event('visibilitychange'));
            }""")
            await wait_for_tracks_ended(page)
            await page.evaluate("""() => {
              Object.defineProperty(document, 'hidden', {configurable: true, value: false});
              document.dispatchEvent(new Event('visibilitychange'));
            }""")
            await assert_no_new_vad_start(page)
            report["checks"].append("hidden page releases microphone and foreground return does not restart")

            # An invalid chat destination rejects the recognized final text,
            # ends the session, and leaves that text for explicit user correction.
            voice.mode = "endpoint"
            parakeet.text = "送信されない最終文です。"
            await page.locator("#llmUrl").fill("not-a-url")
            llm_before = len(llm_calls)
            await start_continuous(page)
            await expect(page.locator("#voiceStatus")).to_contain_text("送信を開始できない", timeout=10000)
            await expect(page.locator("#prompt")).to_have_value(parakeet.text)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert len(llm_calls) == llm_before
            await page.locator("#newChat").click()
            await page.locator("#llmUrl").fill("http://llm.example/v1")
            report["checks"].append("rejected chat submission ends the session and preserves final text")

            await page.locator("#model").fill("fail-model")
            await start_continuous(page)
            await expect(page.locator("#voiceCancel")).to_be_hidden(timeout=10000)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert len(llm_calls) == llm_before + 1
            await page.locator("#model").fill("continuous-browser-test")
            await page.locator("#newChat").click()
            report["checks"].append("LLM failure ends continuous mode without automatic restart")

            # Server-directed backoff applies only to explicitly unaccepted STT
            # requests; subsequent success still produces exactly one chat turn.
            retry_times = []

            async def busy_once(route):
                retry_times.append(asyncio.get_running_loop().time())
                if len(retry_times) == 1:
                    await route.fulfill(status=429, headers={"Retry-After": "1"}, json={"error": "busy"})
                else:
                    await route.continue_()

            await page.route("**/api/stt/transcribe", busy_once)
            done_before = await event_count(page, "llm_done")
            await start_continuous(page)
            await wait_for_event_count(page, "llm_done", done_before + 1, timeout=20000)
            assert len(retry_times) == 2 and retry_times[1] - retry_times[0] >= 1
            await stop_in_cooldown(page, done_before + 1)
            await page.unroute("**/api/stt/transcribe", busy_once)
            await wait_for_tracks_ended(page)
            report["checks"].append("transient STT 429 honors Retry-After and sends one successful turn")

            # Cancellation while recognition is in flight also releases the held
            # stream and cancels the server-side request, without a late send.
            await page.locator("#newChat").click()
            parakeet.delay = 10
            parakeet.entered.clear()
            parakeet.cancelled.clear()
            llm_before = len(llm_calls)
            await start_continuous(page)
            await asyncio.wait_for(parakeet.entered.wait(), 10)
            await page.locator("#voiceCancel").click()
            await asyncio.wait_for(parakeet.cancelled.wait(), 3)
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            assert len(llm_calls) == llm_before
            parakeet.delay = 0
            report["checks"].append("ending during ASR cancels recognition and releases the retained stream")

            # Text-only responses have no audio ACKs but still rearm only after
            # successful completion and the cooldown.
            await page.locator("#audioEnabled").uncheck()
            starts_before = await event_count(page, "vad_start")
            done_before = await event_count(page, "llm_done")
            synthesis_before = len(synthesis_queries)
            await start_continuous(page)
            await wait_for_event_count(page, "llm_done", done_before + 1)
            await wait_for_event_count(page, "vad_start", starts_before + 2)
            assert len(synthesis_queries) == synthesis_before
            await page.locator("#voiceCancel").click()
            await wait_for_tracks_ended(page)
            await assert_no_new_vad_start(page)
            report["checks"].append("successful text-only response also rearms without any TTS requests")

            # Preserve the evidence before reloading resets the browser-side trace.
            last_trace = await trace(page)
            assert not page_errors, page_errors
            starts_before_reload = voice.starts
            await page.reload()
            await expect(page.locator("#voiceContinuous")).to_be_enabled(timeout=15000)
            assert await page.evaluate("window.__streams.length") == 0
            await assert_no_new_vad_start(page, 700)
            assert voice.starts == starts_before_reload, "reload implicitly reconnected the VAD"
            assert not page_errors, page_errors
            report["checks"].append("reload with saved settings does not auto-start continuous capture")
            report["status"] = "passed"
            await browser.close()
            browser = None
    # Preserve the report for any test failure, then re-raise the original error.
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
            "parakeet_requests": len(parakeet.calls),
            "tts_segments": len(synthesis_texts),
            "synthesis_requests": len(synthesis_queries),
            "vad_sessions": voice.starts,
            "vad_closed": voice.closed,
            "microphone_acquisitions": sum(event["type"] == "gum_acquired" for event in last_trace),
            "vad_pcm_frames_observed": sum(event["type"] == "vad_pcm" for event in last_trace),
            "playback_starts": sum(event["type"] == "audio_start" for event in last_trace),
            "played_acks": sum(event["type"] == "played_ack" for event in last_trace),
        }
        report["events"] = [
            event for event in last_trace if event["type"] != "phase_sample"
        ]
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in report.items() if key != "events"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
