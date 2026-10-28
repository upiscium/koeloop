// Each turn owns a player. Cancelling a turn invalidates pending decodes as well as sources.
export class SpeechPlayer {
  constructor(context, played, failed) {
    this.context = context;
    this.played = played;
    this.failed = failed;
    this.sources = new Set();
    this.chain = Promise.resolve();
    this.nextTime = 0;
    this.stopped = false;
  }

  enqueue(event) {
    this.chain = this.chain.then(async () => {
      if (this.stopped) return;
      const bytes = Uint8Array.from(atob(event.wav_base64), (char) => char.charCodeAt(0));
      const buffer = await this.context.decodeAudioData(bytes.buffer);
      if (this.stopped) return;
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.context.destination);
      this.sources.add(source);
      source.onended = () => {
        source.disconnect();
        this.sources.delete(source);
        if (!this.stopped) this.played(event.seq);
      };
      const start = Math.max(this.context.currentTime + 0.03, this.nextTime);
      source.start(start);
      this.nextTime = start + buffer.duration;
    }).catch(() => {
      if (!this.stopped) this.failed("音声を再生できませんでした。ブラウザの音声設定を確認してください。");
    });
  }

  stop() {
    this.stopped = true;
    for (const source of this.sources) {
      source.onended = null;
      try { source.stop(); } catch (_) { /* A source may have just ended. */ }
      source.disconnect();
    }
    this.sources.clear();
  }
}
