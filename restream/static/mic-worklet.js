// Capture mono PCM, but leave every output sample at zero: never monitor the mic.
class MicPCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.buffer = new Float32Array(2048);
    this.used = 0;
    this.total = 0;
    this.limit = options.processorOptions.maxFrames;
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
    for (let index = 0; index < channels[0].length; index++) {
      let value = 0;
      for (const channel of channels) value += channel[index];
      this.buffer[this.used++] = value / channels.length;
      this.total++;
      if (this.used === this.buffer.length) this.flush();
      if (this.total >= this.limit) {
        this.active = false;
        this.flush();
        this.port.postMessage({type: "limit"});
        break;
      }
    }
    return true;
  }
}
registerProcessor("mic-pcm", MicPCMProcessor);
