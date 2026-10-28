// Streaming mono resampling to 16kHz, 100ms PCM blocks, silent monitor output.
// Box-integrated resampling keeps state across render blocks (no per-chunk drift).
class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.width = sampleRate / 16000;
    this.remaining = this.width;
    this.sum = 0;
    this.buffer = new Float32Array(1600);
    this.used = 0;
    this.total = 0;
    this.active = true;
    this.port.onmessage = ({data}) => {
      if (data.type === "flush") {
        this.active = false;
        this.flush();
        this.port.postMessage({type: "flushed"});
      }
    };
  }
  flush() {
    if (!this.used) return;
    const samples = this.buffer.slice(0, this.used);
    this.port.postMessage({type: "samples", samples}, [samples.buffer]);
    this.used = 0;
  }
  process(inputs, outputs) {
    for (const output of outputs) for (const channel of output) channel.fill(0);
    const channels = inputs[0];
    if (!this.active || !channels?.length) return true;
    for (let i = 0; i < channels[0].length && this.active; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[i];
      sample /= channels.length;
      let available = 1;
      while (available > 1e-9 && this.active) {
        const weight = Math.min(available, this.remaining);
        this.sum += sample * weight;
        this.remaining -= weight;
        available -= weight;
        if (this.remaining < 1e-9) {
          this.buffer[this.used++] = this.sum / this.width;
          this.total++;
          this.sum = 0;
          this.remaining = this.width;
          if (this.used === 1600) this.flush();
          if (this.total >= 480000) {
            this.active = false;
            this.flush();
            this.port.postMessage({type: "limit"});
          }
        }
      }
    }
    return true;
  }
}
registerProcessor("voice-pcm", VoiceProcessor);
