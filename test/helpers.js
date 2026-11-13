import { FFT } from '../lib/fft.js';
import { MonoConvolver } from '../lib/convolver.js';

export function maxAbsDiff(a, b) {
  const n = Math.min(a.length, b.length);
  let max = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}

export function rms(a) {
  let sum = 0;
  for (const v of a) sum += v * v;
  return Math.sqrt(sum / a.length);
}

// 教科书式直接卷积，作为小规模真值。
export function naiveConvolution(input, impulse) {
  const out = new Float64Array(input.length + impulse.length - 1);
  for (let i = 0; i < input.length; i++) {
    for (let j = 0; j < impulse.length; j++) {
      out[i + j] += input[i] * impulse[j];
    }
  }
  return Float32Array.from(out);
}

// 单次大 FFT 的重叠保存卷积，作为长冲激响应的独立参考实现。
export function fftConvolution(input, impulse) {
  const n = input.length + impulse.length - 1;
  let fftSize = 1;
  while (fftSize < n) fftSize *= 2;
  const fft = new FFT(fftSize);
  const xr = new Float64Array(fftSize);
  const xi = new Float64Array(fftSize);
  const hr = new Float64Array(fftSize);
  const hi = new Float64Array(fftSize);
  xr.set(input);
  hr.set(impulse);
  fft.transform(xr, xi);
  fft.transform(hr, hi);
  const pr = new Float64Array(fftSize);
  const pi = new Float64Array(fftSize);
  for (let k = 0; k < fftSize; k++) {
    pr[k] = xr[k] * hr[k] - xi[k] * hi[k];
    pi[k] = xr[k] * hi[k] + xi[k] * hr[k];
  }
  fft.inverse(pr, pi);
  return Float32Array.from(pr.subarray(0, n));
}

export function makeImpulse(length) {
  const ir = new Float32Array(length);
  ir[0] = 1;
  return ir;
}

export function makeSine(length, frequency, sampleRate, amplitude = 0.5) {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / sampleRate);
  }
  return out;
}

// 确定性伪随机噪声，保证测试可复现。
export function makeNoise(length, seed = 1234) {
  let state = seed >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    out[i] = (state / 0xffffffff) * 2 - 1;
  }
  return out;
}

export function makeDecayingNoiseIR(length, seed = 77) {
  const noise = makeNoise(length, seed);
  const ir = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    ir[i] = noise[i] * Math.exp(-8 * (i / length));
  }
  return ir;
}

// 按给定块大小序列把同一输入送入新引擎，拼接 process 输出与 flush 输出。
export function runStream(input, impulse, blockSize, chunkSizes, options = {}) {
  const convolver = new MonoConvolver(impulse, {
    sampleRate: 48000,
    blockSize,
    ...options,
  });
  const processed = [];
  let offset = 0;
  let chunkIndex = 0;
  while (offset < input.length) {
    const size = Math.min(chunkSizes[chunkIndex % chunkSizes.length], input.length - offset);
    const chunk = input.subarray(offset, offset + size);
    const out = convolver.process(chunk);
    if (out.length !== chunk.length) {
      throw new Error(`process() returned ${out.length} samples for ${chunk.length} input`);
    }
    processed.push(out);
    offset += size;
    chunkIndex++;
  }
  const tail = convolver.flush();
  const total = processed.reduce((sum, part) => sum + part.length, 0) + tail.length;
  const merged = new Float32Array(total);
  let pos = 0;
  for (const part of processed) {
    merged.set(part, pos);
    pos += part.length;
  }
  merged.set(tail, pos);
  return { merged, tail, processed, convolver };
}
