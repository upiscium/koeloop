// Local STT -> editable draft only. This module deliberately has no send/LLM API.
const RATE = 16000;
const LIMIT_SECONDS = 30;

export function encodeWav(samples) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, RATE, true); view.setUint32(28, RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const sample = Number.isFinite(samples[i]) ? Math.max(-1, Math.min(1, samples[i])) : 0;
    view.setInt16(44 + i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return new Blob([view.buffer], {type: "audio/wav"});
}

export class Dictation {
  constructor({button, cancelButton, status, draft, pendingPanel, pending, insertButton, discardButton, modelSelect, canStart, changed}) {
    Object.assign(this, {button, cancelButton, status, draft, pendingPanel, pending, insertButton, discardButton, modelSelect, canStart, changed});
    this.run = null;
    this.config = null;
    this.state = "idle";
    this.models = [];
    this.backend = "";
    this.message = "音声入力を準備中…";
    button.addEventListener("click", () => this.state === "recording" ? this.finish() : this.start());
    cancelButton.addEventListener("click", () => this.cancel());
    insertButton.addEventListener("click", () => {
      if (!this.busy && this.canStart()) this.insert(this.pending.value);
    });
    discardButton.addEventListener("click", () => { if (!this.busy) this.clearPending(); });
    modelSelect.addEventListener("change", () => {
      if (this.busy || !this.canStart()) { modelSelect.value = this.backend; return; }
      const model = this.models.find((entry) => entry.id === modelSelect.value);
      if (!model?.available) return;
      this.backend = model.id;
      this.config = model;
      this.change("idle", "認識モデルを変更しました。次の録音から適用します。自動送信はしません。");
    });
    window.addEventListener("pagehide", () => this.cancel());
    document.addEventListener("visibilitychange", () => { if (document.hidden && this.busy) this.cancel("画面を離れたため録音・文字起こしを中止しました。"); });
    this.render();
  }
  get busy() { return this.state !== "idle"; }
  unavailable() {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      return "マイクは HTTPS または localhost で利用してください。LAN の HTTP ではブラウザが録音を許可しません。";
    }
    if (!window.AudioContext || !window.AudioWorkletNode) return "このブラウザは音声入力に対応していません。";
    if (!this.config?.available) return this.config?.message || "STT の設定がありません。サーバーを再起動し、画面を再読み込みしてください。";
    return "";
  }
  configure(config, models = []) {
    this.config = config;
    this.models = models;
    this.backend = config?.backend || (models.some((model) => model.id === "default") ? "default" : "");
    const options = models.map((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.engine_label || model.id}${model.available ? "" : "（未導入）"}`;
      option.disabled = !model.available;
      return option;
    });
    if (!models.some((model) => model.id === this.backend)) {
      const fallback = document.createElement("option");
      fallback.value = "";
      fallback.textContent = "サーバーの既定設定";
      options.unshift(fallback);
    }
    this.modelSelect.replaceChildren(...options);
    this.modelSelect.value = this.backend;
    this.change("idle", this.unavailable() || "マイクで録音し、停止すると入力欄へ追加します（最大30秒・自動送信なし）。");
  }
  render() {
    this.button.disabled = this.state === "idle" ? (!this.canStart() || !!this.unavailable()) : this.state !== "recording";
    this.button.textContent = this.state === "recording" ? "■ 文字起こし" : this.busy ? "処理中…" : "マイク";
    this.button.setAttribute("aria-pressed", String(this.state === "recording"));
    this.button.classList.toggle("recording", this.state === "recording");
    this.cancelButton.hidden = !this.busy;
    this.status.textContent = (this.config?.engine_label ? `${this.config.engine_label} — ` : "") + this.message;
    this.pendingPanel.hidden = !this.pending.value;
    this.insertButton.disabled = this.busy || !this.canStart();
    this.discardButton.disabled = this.busy;
    this.modelSelect.disabled = this.busy || !this.canStart();
  }
  change(state, message) {
    this.state = state;
    this.message = message;
    this.render();
    this.changed();
  }
  async release(run) {
    clearInterval(run.timer);
    clearTimeout(run.limitTimer);
    clearTimeout(run.requestTimer);
    run.stream?.getTracks().forEach((track) => track.stop());
    try { run.source?.disconnect(); run.node?.disconnect(); run.node?.port.close(); } catch (_) { /* Already released. */ }
    if (run.context && run.context.state !== "closed") {
      try { await run.context.close(); } catch (_) { /* Page shutdown may already have closed it. */ }
    }
  }
  cancel(message = "録音・文字起こしを破棄しました。入力欄は変更していません。") {
    const run = this.run;
    if (!run) return;
    this.run = null; // Invalidates late permission grants, resampling, and HTTP responses.
    run.controller?.abort();
    run.flush?.();
    void this.release(run);
    this.change("idle", message);
  }
  clearPending() {
    this.pending.value = "";
    this.render();
  }
  reset() {
    this.cancel();
    this.clearPending();
  }
  async start() {
    if (this.busy || !this.canStart()) return;
    const reason = this.unavailable();
    if (reason) return this.change("idle", reason);
    if (this.pending.value) return this.change("idle", "未追加の認識結果があります。追加または破棄してから録音してください。");
    const run = {chunks: [], count: 0, started: performance.now(), backend: this.backend};
    this.run = run;
    this.change("starting", "マイクの使用許可を待っています。自動送信はしません。");
    try {
      run.stream = await navigator.mediaDevices.getUserMedia({audio: {
        channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      }, video: false});
      if (this.run !== run) return await this.release(run);
      try { run.context = new AudioContext({sampleRate: RATE}); } catch (_) { run.context = new AudioContext(); }
      await run.context.audioWorklet.addModule("/static/mic-worklet.js");
      if (this.run !== run) return await this.release(run);
      await run.context.resume();
      if (this.run !== run) return await this.release(run);
      if (run.context.state !== "running") throw new Error("音声入力を開始できませんでした。ブラウザの音声設定を確認してください。");
      run.rate = run.context.sampleRate;
      run.node = new AudioWorkletNode(run.context, "mic-pcm", {numberOfInputs: 1, numberOfOutputs: 1,
        outputChannelCount: [1], processorOptions: {maxFrames: run.rate * LIMIT_SECONDS}});
      run.node.port.onmessage = ({data}) => {
        if (this.run !== run) return;
        if (data.type === "samples") {
          const remaining = run.rate * LIMIT_SECONDS - run.count;
          const samples = data.samples.subarray(0, remaining);
          run.chunks.push(samples);
          run.count += samples.length;
        } else if (data.type === "flushed") run.flush?.();
        else if (data.type === "limit") void this.finish();
      };
      run.source = run.context.createMediaStreamSource(run.stream);
      run.source.connect(run.node);
      run.node.connect(run.context.destination); // Worklet output is always zero.
      run.started = performance.now();
      this.change("recording", "録音中 0 / 30秒。■ 文字起こしで停止します。自動送信なし。");
      run.timer = setInterval(() => {
        if (this.run === run && this.state === "recording") {
          this.message = `録音中 ${Math.min(30, Math.floor((performance.now() - run.started) / 1000))} / 30秒。■ 文字起こしで停止します。自動送信なし。`;
          this.render();
        }
      }, 250);
      run.limitTimer = setTimeout(() => this.finish(), LIMIT_SECONDS * 1000);
      for (const track of run.stream.getAudioTracks()) track.addEventListener("ended", () => {
        if (this.run === run && this.state === "recording") this.cancel("マイクとの接続が切れました。録音し直してください。");
      });
    } catch (error) {
      if (this.run !== run) return;
      const messages = {NotAllowedError: "マイクの使用が許可されていません。ブラウザのサイト設定を確認してください。",
        NotFoundError: "マイクが見つかりません。", NotReadableError: "マイクを利用できません。他のアプリの使用状況を確認してください。"};
      this.cancel(messages[error.name] || "録音を開始できませんでした。ブラウザとマイクの設定を確認してください。");
    }
  }
  async finish() {
    const run = this.run;
    if (!run || this.state !== "recording") return;
    this.change("encoding", "録音を停止して音声を準備しています…");
    clearInterval(run.timer);
    clearTimeout(run.limitTimer);
    try {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("録音を取り出せませんでした。録音し直してください。")), 2000);
        run.flush = () => { clearTimeout(timer); resolve(); };
        run.node.port.postMessage({type: "flush"});
      });
      await this.release(run);
      if (this.run !== run) return;
      if (run.count / run.rate < 0.5) throw new Error("0.5秒以上録音してください。");
      const pcm = new Float32Array(run.count);
      let offset = 0;
      for (const chunk of run.chunks) { pcm.set(chunk, offset); offset += chunk.length; }
      run.chunks = [];
      let samples = pcm;
      if (run.rate !== RATE) {
        const offline = new OfflineAudioContext(1, Math.min(RATE * LIMIT_SECONDS, Math.round(pcm.length * RATE / run.rate)), RATE);
        const buffer = offline.createBuffer(1, pcm.length, run.rate);
        buffer.copyToChannel(pcm, 0);
        const source = offline.createBufferSource();
        source.buffer = buffer;
        source.connect(offline.destination);
        source.start();
        samples = (await offline.startRendering()).getChannelData(0);
      }
      if (this.run !== run) return;
      this.change("transcribing", "ローカル STT で文字起こし中…（LLM には送信しません）");
      run.controller = new AbortController();
      run.requestTimer = setTimeout(() => { run.timedOut = true; run.controller.abort(); }, 95000);
      const headers = {"Content-Type": "audio/wav"};
      if (run.backend) headers["X-STT-Backend"] = run.backend;
      const response = await fetch("/api/stt/transcribe", {method: "POST", headers,
        body: encodeWav(samples), signal: run.controller.signal});
      const result = await response.json();
      if (this.run !== run) return;
      if (!response.ok) throw new Error(result.error || "文字起こしに失敗しました。");
      if (typeof result.text !== "string") throw new Error("文字起こしの応答が不正です。");
      this.run = null;
      clearTimeout(run.requestTimer);
      this.change("idle", "文字起こしが完了しました。");
      this.insert(result.text);
    } catch (error) {
      if (this.run === run) this.cancel(run.timedOut ? "文字起こしがタイムアウトしました。" : error.message);
    }
  }
  insert(text) {
    text = text.trim();
    if (!text) return this.change("idle", "認識できる音声がありませんでした。入力欄は変更していません。");
    const current = this.draft.value; // Preserve edits made while recording/transcribing.
    const separator = current && !/\s$/.test(current) ? "\n" : "";
    if ((current + separator + text).length > this.draft.maxLength) {
      this.pending.value = text;
      return this.change("idle", "入力欄の文字数上限を超えるため、結果を下に残しました。下書きを整理してから追加してください。");
    }
    this.draft.value = current + separator + text;
    this.clearPending();
    // No submit(), requestSubmit(), key events, WebSocket, or LLM calls here.
    this.change("idle", "文字起こしを入力欄に追加しました。内容を確認・修正してください（自動送信なし）。");
  }
}
