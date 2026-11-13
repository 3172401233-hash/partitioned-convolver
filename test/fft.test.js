import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FFT, isPowerOfTwo } from '../lib/fft.js';
import { makeNoise } from './helpers.js';

test('FFT 往返变换保持信号不变', () => {
  const size = 64;
  const fft = new FFT(size);
  const original = makeNoise(size, 99);
  const real = Float64Array.from(original);
  const imag = new Float64Array(size);
  fft.transform(real, imag);

  // 能量守恒（Parseval）。
  let timeEnergy = 0;
  let freqEnergy = 0;
  for (let i = 0; i < size; i++) {
    timeEnergy += original[i] * original[i];
    freqEnergy += real[i] * real[i] + imag[i] * imag[i];
  }
  assert.ok(Math.abs(freqEnergy / size - timeEnergy) < 1e-10);

  fft.inverse(real, imag);
  let maxError = 0;
  for (let i = 0; i < size; i++) {
    maxError = Math.max(maxError, Math.abs(real[i] - original[i]));
    assert.ok(Math.abs(imag[i]) < 1e-14);
  }
  assert.ok(maxError < 1e-12, `round-trip error ${maxError}`);
});

test('FFT 与朴素 DFT 结果一致', () => {
  const size = 32;
  const fft = new FFT(size);
  const signal = makeNoise(size, 5);
  const real = Float64Array.from(signal);
  const imag = new Float64Array(size);
  fft.transform(real, imag);

  for (let k = 0; k < size; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < size; t++) {
      const angle = (-2 * Math.PI * k * t) / size;
      re += signal[t] * Math.cos(angle);
      im += signal[t] * Math.sin(angle);
    }
    assert.ok(Math.abs(real[k] - re) < 1e-10);
    assert.ok(Math.abs(imag[k] - im) < 1e-10);
  }
});

test('单位脉冲的频谱恒为 1，直流的能量集中在 0 号 bin', () => {
  const size = 16;
  const fft = new FFT(size);
  const real = new Float64Array(size);
  const imag = new Float64Array(size);
  real[0] = 1;
  fft.transform(real, imag);
  for (let k = 0; k < size; k++) {
    assert.ok(Math.abs(real[k] - 1) < 1e-14);
    assert.ok(Math.abs(imag[k]) < 1e-14);
  }

  const dcReal = new Float64Array(size).fill(3);
  const dcImag = new Float64Array(size);
  fft.transform(dcReal, dcImag);
  assert.ok(Math.abs(dcReal[0] - 3 * size) < 1e-12);
  for (let k = 1; k < size; k++) {
    assert.ok(Math.abs(dcReal[k]) < 1e-12);
  }
});

test('非 2 的幂尺寸直接报错', () => {
  assert.ok(!isPowerOfTwo(0));
  assert.ok(!isPowerOfTwo(1000));
  assert.ok(isPowerOfTwo(2048));
  assert.throws(() => new FFT(0), /power of two/);
  assert.throws(() => new FFT(1023), /power of two/);
});
