// Explicit opt-in voice turn: VAD -> Parakeet -> one submission. No partial ASR.
import { encodeWav } from "./dictation.js";

export function capturedRange(chunks, start, end) {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > 480000
      || chunks.reduce((count, chunk) => count + chunk.length, 0) < end) {
    throw new Error("録音範囲を確認できません。サーバーを再起動して録音し直してください。");
  }
  // Preserve very short valid speech: the batch API requires >=0.5s WAV.
  const samples = new Float32Array(Math.max(8000, end - start));
  let offset = 0, position = 0;
  for (const chunk of chunks) {
    const from = Math.max(0, start - position), to = Math.min(chunk.length, end - position);
    if (to > from) { samples.set(chunk.subarray(from, to), offset); offset += to - from; }
    position += chunk.length;
    if (position >= end) break;
  }
  return samples;
}

export function pcm16(samples) {
  const view = new DataView(new ArrayBuffer(samples.length * 2));
  for (let i = 0; i < samples.length; i++) {
    const sample = Number.isFinite(samples[i]) ? Math.max(-1, Math.min(1, samples[i])) : 0;
    view.setInt16(i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return view.buffer;
}

export class VoiceTurn {
  constructor({button, continuousButton, cancelButton, status, draft, silenceInput, preInput, postInput, thresholdInput, bargeInput, hint, canStart, changed, prepareAudio, committed, stopReply, interruptReply}) {
    Object.assign(this, {button, continuousButton, cancelButton, status, draft, silenceInput, preInput, postInput, thresholdInput, bargeInput, hint, canStart, changed, prepareAudio, committed, stopReply, interruptReply});
    this.run = null;
    this.session = null;
    this.config = null;
    this.finalAvailable = false;
    this.message = "発話終了を検出してParakeetで認識し、LLMへ送信します。";
    button.addEventListener("click", () => this.run?.recording ? this.end() : this.start());
    continuousButton.addEventListener("click", () => this.start({continuous: true}));
    cancelButton.addEventListener("click", () => this.cancel());
    window.addEventListener("pagehide", () => this.cancel());
    document.addEventListener("visibilitychange", () => { if (document.hidden) this.cancel(); });
    this.render();
  }
  get busy() { return !!this.run || !!this.session; }
  canSubmit(session) {
    return !this.run && (session ? this.session === session && session.phase === "reply" : !this.session);
  }
  configure(config, backends = [], preferences = {}) {
    this.config = config;
    this.finalAvailable = backends.some((entry) => entry.id === "parakeet" && entry.available);
    const integer = (value, min, max, fallback) => Number.isInteger(value) && value >= min && value <= max ? value : fallback;
    this.silenceInput.value = integer(preferences.voiceSilenceMs, 500, 3000, 1000);
    this.preInput.value = integer(preferences.voicePreMs, 0, 500, 200);
    this.postInput.value = integer(preferences.voicePostMs, 0, 500, 200);
    this.thresholdInput.value = Number.isFinite(preferences.voiceThreshold) && preferences.voiceThreshold >= 0.1 && preferences.voiceThreshold <= 0.9 ? preferences.voiceThreshold : 0.5;
    this.bargeInput.checked = preferences.voiceBargeIn === true;
    this.hint.textContent = this.finalAvailable
      ? "VADは発話区間だけを検出し、Parakeetだけが文字起こしします。途中の文字は表示しません。言いよどみで切れる場合は無音待ちを長くしてください。"
      : "Parakeetが未導入です。音声会話は nix develop .#voice で起動してください。";
    this.render();
  }
  render() {
    const listening = this.run?.recording && !this.run.monitor;
    this.button.disabled = this.busy ? !listening : !this.canStart() || !this.config?.available || !this.finalAvailable;
    this.button.textContent = this.busy ? (listening ? "■ 話し終わり・送信" : this.run?.monitor ? "割り込みを待機中…" : "聞き取り停止中…") : "音声で会話";
    this.continuousButton.disabled = this.busy || !this.canStart() || !this.config?.available || !this.finalAvailable;
    this.continuousButton.hidden = !!this.session;
    this.button.classList.toggle("recording", !!listening);
    this.cancelButton.hidden = !this.busy;
    this.cancelButton.textContent = this.session ? "会話を終了" : "音声会話を破棄";
    this.status.textContent = !this.config?.available ? this.config?.message || "音声会話の設定を確認中…"
      : !this.finalAvailable ? "音声会話にはParakeetが必要です。nix develop .#voice で起動してください。" : this.message;
  }
  update(message) {
    this.message = message;
    this.render();
    this.changed();
  }
  release(run) {
    return run.releasing ||= this.releaseResources(run);
  }
  pauseStream(run) {
    for (const track of run.stream?.getTracks() || []) {
      if (run.session && this.session === run.session) track.enabled = false;
      else track.stop();
    }
  }
  async releaseResources(run) {
    clearTimeout(run.timer);
    clearTimeout(run.timeout);
    clearTimeout(run.refinementTimer);
    run.resolveReady?.();
    run.resolveFlush?.();
    this.pauseStream(run);
    try { run.source?.disconnect(); run.node?.disconnect(); run.node?.port.close(); } catch (_) { /* Already closed. */ }
    if (run.context && run.context.state !== "closed") {
      try { await run.context.close(); } catch (_) { /* Page shutdown. */ }
    }
    if (run.ws && run.ws.readyState < WebSocket.CLOSING) run.ws.close();
  }
  cancel(message) {
    const run = this.run;
    const session = this.session;
    if (!run && !session) return;
    // Invalidate ownership before stopping the reply: its finish callback must
    // never rearm a session that the user has ended.
    this.session = null;
    this.run = null;
    if (session) {
      clearTimeout(session.timer);
      session.reply = null;
      session.stream?.getTracks().forEach((track) => track.stop());
    }
    if (run) {
      run.controller?.abort();
      if (run.ws?.readyState === WebSocket.OPEN) run.ws.send(JSON.stringify({type: "cancel"}));
      this.draft.value = run.base;
      void this.release(run);
    }
    this.draft.readOnly = false;
    if (session) this.stopReply();
    this.update(message || (session ? "連続会話を終了しました。マイクを解放しました。"
      : "音声会話を破棄しました。LLMには送信していません。"));
  }
  resume(session, message = "返答が完了しました。反響が収まるのを待って聞き取りを再開します。") {
    if (this.session !== session) return;
    session.phase = "cooldown";
    this.update(message);
    session.timer = setTimeout(() => {
      if (this.session !== session) return;
      if (document.hidden || !this.canStart() || this.draft.value.trim()) {
        return this.cancel("聞き取りを安全に再開できないため、連続会話を終了しました。");
      }
      void this.start({session});
    }, 500);
  }
  replyFinished(session, success, reply) {
    if (!session) {
      if (!this.session) this.update(success ? "返答が完了しました。次の発話は「音声で会話」を押してください。"
        : "返答処理が停止しました。上の状態表示を確認してください。");
      return;
    }
    if (this.session !== session || session.reply !== reply) return;
    if (!success) return this.cancel("返答が停止・失敗したため、連続会話を終了しました。上の状態表示を確認してください。");
    session.reply = null;
    clearTimeout(session.timer);
    const run = this.run;
    if (run?.monitor && run.reply === reply) {
      if (!run.captureStarted) {
        // No onset exists yet. Retire pending setup rather than accidentally
        // starting strict-threshold capture immediately after playback ends.
        this.run = null;
        session.phase = "cooldown";
        this.update("返答が完了しました。割り込み準備を閉じて、聞き取りを再開します。");
        void this.release(run).then(() => {
          if (this.session === session && !this.run && !session.reply) this.resume(session);
        });
        return;
      }
      // The microphone already holds the next utterance's onset. Do not flush,
      // disable, or replace this capture merely because playback just ended.
      run.monitor = false;
      run.reply = null;
      session.phase = "listening";
      this.update("返答が完了しました。引き続き聞いています。話し始めてください。");
      return;
    }
    this.resume(session);
  }
  interrupt(run) {
    if (!run.monitor) return true;
    const {session, reply} = run;
    if (this.run !== run || this.session !== session || session.reply !== reply) return false;
    // Revoke the reply lease first. Its delayed done/error cannot end this
    // listening turn, and stopping playback must not discard captured speech.
    session.reply = null;
    run.monitor = false;
    run.reply = null;
    session.phase = "listening";
    if (!this.interruptReply(reply)) {
      this.cancel("割り込み対象の返答を確認できないため、連続会話を終了しました。");
      return false;
    }
    this.update("割り込みを検出して返答を中断しました。聞いています。話を続けてください。");
    return true;
  }
  armMonitor(session, reply, delay = 0) {
    if (this.session !== session || session.reply !== reply || session.phase !== "reply") return;
    clearTimeout(session.timer);
    session.timer = setTimeout(() => {
      if (this.session !== session || session.reply !== reply || this.run) return;
      if (document.hidden) return this.cancel();
      void this.start({session, monitor: true, reply});
    }, delay);
  }
  pump(run) {
    if (this.run !== run || !run.ready || run.inFlight) return;
    if (run.queue.length) {
      run.inFlight = true;
      run.ws.send(run.queue.shift());
    } else if (run.ended && !run.endSent) {
      run.endSent = true;
      run.ws.send(JSON.stringify({type: "end"}));
    }
  }
  async start({continuous = false, session = null, monitor = false, reply = null} = {}) {
    const canListen = monitor ? session && this.session === session && session.reply === reply && session.phase === "reply" : this.canStart();
    if (this.run || (this.session && this.session !== session) || !canListen || !this.config?.available || !this.finalAvailable) return;
    if (session && this.session !== session) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      return this.update("音声会話には HTTPS または localhost と対応ブラウザが必要です。");
    }
    if (this.draft.value.trim()) return this.update("誤送信を防ぐため、下書きを送信または消してから音声会話を開始してください。");
    const number = (input) => input.value.trim() === "" ? NaN : Number(input.value);
    const settings = {silence_ms: number(this.silenceInput), pre_ms: number(this.preInput),
      post_ms: number(this.postInput), threshold: number(this.thresholdInput)};
    if (!Number.isInteger(settings.silence_ms) || settings.silence_ms < 500 || settings.silence_ms > 3000
        || ![settings.pre_ms, settings.post_ms].every(value => Number.isInteger(value) && value >= 0 && value <= 500)
        || !Number.isFinite(settings.threshold) || settings.threshold < 0.1 || settings.threshold > 0.9) {
      return this.update("VAD設定は、無音待ち500～3000ms、前後余白0～500ms、しきい値0.1～0.9で指定してください。");
    }
    if (monitor) settings.threshold = Math.max(settings.threshold, 0.65);
    if (continuous) this.session = session = {phase: "preparing", stream: null, bargeIn: this.bargeInput.checked};
    if (session && !monitor) session.phase = "listening";
    const run = {base: this.draft.value, queue: [], captured: [], inFlight: false, settings, session, monitor, reply};
    this.run = run;
    this.draft.readOnly = true;
    this.update(monitor ? "返答中の割り込み検出を準備しています…" : "音声会話を準備中…発話終了後に設定済みLLMへ送信します。");
    try {
      await this.prepareAudio(); // Unlock TTS playback within the explicit user gesture.
      if (this.run !== run) return;
      // Load the resident model BEFORE acquiring the microphone. Otherwise a
      // cold model can consume the beginning of speech while capture is not ready.
      const ready = new Promise((resolve) => { run.resolveReady = resolve; });
      run.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/vad`);
      run.timeout = setTimeout(() => this.cancel("音声会話の準備・認識がタイムアウトしました。未送信です。"), 45000);
      run.ws.onopen = () => { if (this.run === run) run.ws.send(JSON.stringify({type: "start", ...run.settings})); };
      run.ws.onmessage = ({data}) => {
        if (this.run !== run || run.finalizing) return;
        try {
          const event = JSON.parse(data);
          if (event.type === "ready") {
            run.ready = true;
            run.resolveReady();
          } else if (event.type === "ack") {
            run.inFlight = false;
            this.pump(run);
          } else if (event.type === "state") {
            if (!["waiting", "speaking"].includes(event.state)) throw new Error("invalid VAD state");
            if (run.monitor && event.state === "speaking" && !this.interrupt(run)) return;
            this.update(run.monitor ? "返答中も聞いています（割り込み待ち）。話し始めると返答を中断します。"
              : event.state === "speaking" ? "聞いています（発話中）。無音が続くと認識・送信します。"
              : "聞いています（発話待ち）。話し始めてください。");
          } else if (event.type === "end") {
            if (typeof event.speech !== "boolean") throw new Error("invalid VAD endpoint");
            if (run.monitor && event.speech && !this.interrupt(run)) return;
            void this.commit(run, event);
          } else if (event.type === "error") this.cancel(event.message);
          else if (event.type === "cancelled") this.cancel();
        } catch (_) { this.cancel("発話検出の応答を処理できませんでした。未送信です。"); }
      };
      run.ws.onerror = () => {
        if (this.run === run && !run.finalizing) this.cancel("発話検出に接続できませんでした。未送信です。");
      };
      run.ws.onclose = () => { if (this.run === run && !run.finalizing) this.cancel("認識完了前に接続が切れました。未送信です。"); };
      await ready;
      if (this.run !== run) return;
      this.update(run.monitor ? "返答中の割り込み検出を準備しています…" : "認識器の準備ができました。マイクの使用許可を待っています…");
      // Prepare the capture context/worklet before the device starts delivering audio.
      try { run.context = new AudioContext({sampleRate: 16000}); } catch (_) { run.context = new AudioContext(); }
      await run.context.audioWorklet.addModule("/static/voice-worklet.js");
      if (this.run !== run) return await this.release(run);
      await run.context.resume();
      if (this.run !== run) return await this.release(run);
      if (run.context.state !== "running") throw new Error("マイク処理を開始できませんでした。");
      run.node = new AudioWorkletNode(run.context, "voice-pcm", {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]});
      run.node.port.onmessage = ({data}) => {
        if (this.run !== run || run.finalizing) return;
        if (data.type === "samples") {
          if (run.queue.length >= 20) return this.cancel("認識が音声入力に追いつきません。録音を中止しました。未送信です。");
          run.captured.push(data.samples);
          run.queue.push(pcm16(data.samples));
          this.pump(run);
        } else if (data.type === "flushed") run.resolveFlush?.();
        else if (data.type === "limit") { run.limited = true; void this.end(); }
      };
      const stream = session?.stream || await navigator.mediaDevices.getUserMedia({audio: {
        channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      }, video: false});
      // A permission prompt may outlive cancellation. Never adopt a late stream.
      if (this.run !== run) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      run.stream = stream;
      if (session) session.stream = stream;
      for (const track of stream.getTracks()) {
        if (track.readyState !== "live") throw new Error("microphone ended");
        track.enabled = true;
      }
      run.source = run.context.createMediaStreamSource(run.stream);
      run.source.connect(run.node);
      run.node.connect(run.context.destination);
      run.captureStarted = true;
      run.recording = true;
      run.timer = setTimeout(() => { run.limited = true; void this.end(); }, 30000);
      clearTimeout(run.timeout);
      run.timeout = setTimeout(() => this.cancel("認識がタイムアウトしました。未送信です。"), 60000);
      this.update(run.monitor ? "返答中も聞いています（割り込み待ち）。話し始めると返答を中断します。"
        : "聞いています（発話待ち）。話し始めてください。");
      if (!session || !session.watchingTracks) {
        if (session) session.watchingTracks = true;
        for (const track of stream.getTracks()) track.addEventListener("ended", () => {
          if (session ? this.session === session : this.run === run && run.recording) {
            this.cancel("マイクとの接続が切れました。音声会話を終了しました。");
          }
        });
      }
    } catch (_) {
      if (this.run === run) this.cancel("録音を開始できません。マイクの許可・HTTPS・接続設定を確認してください。未送信です。");
    }
  }
  async end() {
    const run = this.run;
    if (!run?.recording) return;
    run.recording = false;
    this.update("録音を停止して発話を確定しています…");
    clearTimeout(run.timer);
    try {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("flush timeout")), 2000);
        run.resolveFlush = () => { clearTimeout(timer); resolve(); };
        run.node.port.postMessage({type: "flush"});
      });
      this.pauseStream(run);
      if (this.run !== run || run.finalizing) return;
      run.ended = true;
      this.pump(run);
    } catch (_) { if (this.run === run) this.cancel("録音を確定できませんでした。未送信です。"); }
  }
  settle(run, message, text = "") {
    if (this.run !== run) return;
    this.run = null;
    this.draft.readOnly = !!this.session;
    this.draft.value = text;
    this.update(message);
  }
  async commit(run, endpoint) {
    if (this.run !== run || run.finalizing) return;
    run.finalizing = true;
    run.recording = false;
    await this.release(run); // No capture graph or enabled mic during ASR/reply.
    if (this.run !== run) return; // Cancel/new conversation wins over delayed cleanup.
    if (run.monitor && !endpoint.speech) {
      this.settle(run, "発話を検出していません。返答を続けながら割り込み待ちの区間を更新します。");
      this.armMonitor(run.session, run.reply, 100);
      return;
    }
    if (run.session && !endpoint.speech) {
      this.settle(run, "発話を検出していません。送信せず、聞き取りを続けます。");
      return this.resume(run.session, "発話待ちです。送信せず、聞き取りの区間を更新します。");
    }
    if (run.limited) {
      if (run.session) this.cancel("30秒上限で停止しました。途中の可能性があるため認識・送信せず、連続会話を終了しました。");
      else this.settle(run, "30秒上限で停止しました。途中の可能性があるため認識・送信しません。録音し直してください。");
      return;
    }
    if (!endpoint.speech) return this.settle(run, "発話を検出できませんでした。LLMには送信しません。");
    try {
      const body = encodeWav(capturedRange(run.captured, endpoint.start_sample, endpoint.end_sample));
      run.captured = [];
      run.controller = new AbortController();
      run.refinementTimer = setTimeout(() => { run.timedOut = true; run.controller.abort(); }, 90000);
      this.update("認識しています（Parakeet）。まだLLMへ送信していません。");
      let response;
      for (let attempt = 0; attempt < 6; attempt++) {
        response = await fetch("/api/stt/transcribe", {method: "POST",
          headers: {"Content-Type": "audio/wav", "X-STT-Backend": "parakeet"},
          body, signal: run.controller.signal});
        // Only an explicitly unaccepted (429) request may be retried.
        if (response.status !== 429 || attempt === 5) break;
        const retryAfter = Number(response.headers.get("Retry-After") || "1");
        const delay = Number.isFinite(retryAfter) ? Math.max(250, Math.min(5000, retryAfter * 1000)) : 1000;
        await new Promise((resolve) => setTimeout(resolve, delay));
        if (this.run !== run) return;
      }
      const result = await response.json();
      if (this.run !== run) return;
      if (!response.ok) throw new Error(result.error || "音声認識に失敗しました。");
      if (typeof result.text !== "string" || !result.text.trim() || result.text.length > this.draft.maxLength) {
        throw new Error("認識結果が空、または入力上限を超えています。");
      }
      // Only this final text reaches the composer, immediately before creating
      // the normal user chat bubble. No partial/draft transcript is displayed.
      this.settle(run, "発話を認識しました。LLMへ送信します。", result.text.trim());
      if (run.session) {
        run.session.phase = "reply";
        run.session.reply = {};
        this.update(run.session.bargeIn ? "返答を生成・読み上げています。割り込み検出を準備します。"
          : "返答を生成・読み上げています。割り込みなしのため、聞き取りは停止中です。");
      }
      const reply = run.session?.reply;
      const accepted = await this.committed(run.session, reply);
      if (run.session && this.session === run.session && !accepted) {
        this.cancel("送信を開始できないため、連続会話を終了しました。入力欄と接続設定を確認してください。");
      } else if (accepted && run.session?.bargeIn) {
        this.armMonitor(run.session, reply);
      }
    } catch (error) {
      if (this.run === run) {
        const message = `${run.timedOut ? "音声認識がタイムアウトしました。" : error.message} 自動送信しません。録音し直してください。`;
        if (run.session) this.cancel(message);
        else this.settle(run, message);
      } else if (run.session && this.session === run.session) {
        this.cancel("送信処理でエラーが出たため、連続会話を終了しました。上の状態表示を確認してください。");
      }
    } finally {
      clearTimeout(run.refinementTimer);
    }
  }
}
