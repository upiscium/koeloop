import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { encodeWav } from "../restream/static/dictation.js";
import { capturedRange, pcm16 } from "../restream/static/voice-turn.js";

function processor(maxFrames) {
  let Type;
  const messages = [];
  const context = vm.createContext({Float32Array,
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage: (data) => messages.push(data)}; } },
    registerProcessor: (_name, value) => { Type = value; },
  });
  vm.runInContext(readFileSync(new URL("../restream/static/mic-worklet.js", import.meta.url), "utf8"), context);
  return {instance: new Type({processorOptions: {maxFrames}}), messages};
}

test("WAV is mono 16kHz PCM16 with clipped finite samples", async () => {
  const blob = encodeWav(new Float32Array([-2, -1, 0, 1, 2, NaN]));
  const view = new DataView(await blob.arrayBuffer());
  assert.equal(blob.type, "audio/wav");
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 12);
  assert.deepEqual(Array.from({length: 6}, (_, i) => view.getInt16(44 + i * 2, true)), [-32768, -32768, 0, 32767, 32767, 0]);
});

test("worklet never monitors the mic and bounds capture", () => {
  const {instance, messages} = processor(16000);
  for (let i = 0; i < 200; i++) {
    const output = new Float32Array(128).fill(1);
    instance.process([[new Float32Array(128).fill(0.5)]], [[output]]);
    assert(output.every((value) => value === 0));
  }
  assert.equal(messages.filter((m) => m.type === "samples").reduce((n, m) => n + m.samples.length, 0), 16000);
  assert.equal(messages.filter((m) => m.type === "limit").length, 1);
});

test("flush preserves the short final buffer and stops further capture", () => {
  const {instance, messages} = processor(16000);
  instance.process([[new Float32Array(128).fill(0.5), new Float32Array(128).fill(-0.5)]], [[new Float32Array(128)]]);
  instance.port.onmessage({data: {type: "flush"}});
  instance.process([[new Float32Array(128).fill(1)]], [[new Float32Array(128)]]);
  const chunks = messages.filter((m) => m.type === "samples");
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].samples.length, 128);
  assert(chunks[0].samples.every((value) => value === 0));
  assert.equal(messages.filter((m) => m.type === "flushed").length, 1);
});

function voiceProcessor(rate) {
  let Type;
  const messages = [];
  const context = vm.createContext({Float32Array, sampleRate: rate,
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage: (data) => messages.push(data)}; } },
    registerProcessor: (_name, value) => { Type = value; },
  });
  vm.runInContext(readFileSync(new URL("../restream/static/voice-worklet.js", import.meta.url), "utf8"), context);
  return {instance: new Type(), messages};
}

test("voice PCM encoding is little-endian and clips safely", () => {
  const view = new DataView(pcm16(new Float32Array([-1, 0, 1, NaN])));
  assert.deepEqual(Array.from({length: 4}, (_, i) => view.getInt16(i * 2, true)), [-32768, 0, 32767, 0]);
});

test("final recognition includes the accepted utterance, not queued trailing audio", () => {
  const chunks = [new Float32Array(4800).fill(0.25), new Float32Array(4800).fill(0.5), new Float32Array(1600).fill(1)];
  const result = capturedRange(chunks, 1600, 9600);
  assert.equal(result.length, 8000);
  assert(result.subarray(0, 3200).every((value) => value === 0.25));
  assert(result.subarray(3200).every((value) => value === 0.5));
  for (const count of [undefined, -1, 0, 12000, 480001, 8000.5]) {
    assert.throws(() => capturedRange(chunks, 0, count));
  }
  assert.throws(() => capturedRange(chunks, -1, 100));
  assert.throws(() => capturedRange(chunks, 1000, 500));
  const short = capturedRange(chunks, 1600, 3200);
  assert.equal(short.length, 8000);
  assert(short.subarray(0, 1600).every(value => value === 0.25));
  assert(short.subarray(1600).every(value => value === 0));
});

for (const rate of [16000, 44100, 48000]) test(`stream resampler retains duration and sample values at ${rate}Hz`, () => {
  const {instance, messages} = voiceProcessor(rate);
  for (let i = 0; i < rate; i += 128) {
    const output = new Float32Array(128).fill(1);
    instance.process([[new Float32Array(Math.min(128, rate - i)).fill(0.25)]], [[output]]);
    assert(output.every((value) => value === 0));
  }
  instance.port.onmessage({data: {type: "flush"}});
  const chunks = messages.filter((event) => event.type === "samples");
  assert.equal(chunks.reduce((count, chunk) => count + chunk.samples.length, 0), 16000);
  assert(chunks.every((chunk) => chunk.samples.every((value) => Math.abs(value - 0.25) < 1e-6)));
});

test("voice capture stops at 30 seconds", () => {
  const {instance, messages} = voiceProcessor(16000);
  for (let i = 0; i < 500000; i += 128) instance.process([[new Float32Array(128)]], [[new Float32Array(128)]]);
  assert.equal(messages.filter((event) => event.type === "samples").reduce((count, event) => count + event.samples.length, 0), 480000);
  assert.equal(messages.filter((event) => event.type === "limit").length, 1);
});
