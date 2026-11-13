import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MonoConvolver, StereoConvolver } from '../lib/convolver.js';
import {
  fftConvolution,
  naiveConvolution,
  makeDecayingNoiseIR,
  makeImpulse,
  makeNoise,
  makeSine,
  maxAbsDiff,
  rms,
  runStream,
} from './helpers.js';

const SR = 48000;

// 核心不变量：完整输出去掉前 B 个延迟样本后等于线性卷积真值。
function assertMatchesLinearConvolution(merged, blockSize, input, impulse, tolerance) {
  const reference = fftConvolution(input, impulse);
  assert.equal(merged.length, blockSize + reference.length);
  const aligned = merged.subarray(blockSize);
  const error = maxAbsDiff(aligned, reference);
  assert.ok(
    error < tolerance,
    `streamed convolution error ${error} exceeds tolerance ${tolerance}`,
  );
  return error;
}

test('单位脉冲冲激响应：输出即延迟后的输入', () => {
  const blockSize = 64;
  const input = makeSine(500, 440, SR, 0.7);
  const ir = makeImpulse(1);
  const { merged, tail } = runStream(input, ir, blockSize, [127, 33, 256, 1]);

  assert.equal(tail.length, blockSize);
  const aligned = merged.subarray(blockSize, blockSize + input.length);
  const error = maxAbsDiff(aligned, input);
  assert.ok(error < 1e-6, `identity error ${error}`);
  // 输入结束后只剩零样本。
  for (const v of merged.subarray(blockSize + input.length)) {
    assert.equal(v, 0);
  }
});

test('同一段音频无论怎么分块，输出逐样本完全一致', () => {
  const blockSize = 128;
  const input = makeNoise(4001, 42);
  const impulse = makeDecayingNoiseIR(700, 7);
  const chunkPlans = [
    [1],
    [7],
    [blockSize],
    [blockSize * 3],
    [3, 5, 8, 13, 21],
    [1000, 1, 1, 2, 999],
    [17, 257, 300, 64, 2],
  ];

  const baseline = runStream(input, impulse, blockSize, chunkPlans[0]).merged;
  for (let plan = 1; plan < chunkPlans.length; plan++) {
    const result = runStream(input, impulse, blockSize, chunkPlans[plan]).merged;
    assert.equal(result.length, baseline.length);
    // 分块只影响缓冲节奏，不允许任何数值差异。
    assert.ok(
      maxAbsDiff(result, baseline) === 0,
      `chunk plan ${plan} produced different samples`,
    );
  }
});

test('process 每次返回与输入等长的数据，且结果匹配朴素卷积', () => {
  const blockSize = 32;
  const input = makeSine(300, 1000, SR, 0.4);
  const impulse = makeDecayingNoiseIR(90, 11);
  const convolver = new MonoConvolver(impulse, { sampleRate: SR, blockSize });

  const chunkSizes = [5, 50, 17, 64, 3];
  const outputs = [];
  let offset = 0;
  let plan = 0;
  while (offset < input.length) {
    const size = Math.min(chunkSizes[plan % chunkSizes.length], input.length - offset);
    const out = convolver.process(input.subarray(offset, offset + size));
    assert.equal(out.length, size);
    assert.ok(out instanceof Float32Array);
    outputs.push(out);
    offset += size;
    plan++;
  }
  const tail = convolver.flush();
  assert.equal(tail.length, blockSize + impulse.length - 1);

  const merged = new Float32Array(input.length + tail.length);
  let pos = 0;
  for (const part of outputs) {
    merged.set(part, pos);
    pos += part.length;
  }
  merged.set(tail, pos);

  const reference = naiveConvolution(input, impulse);
  const error = maxAbsDiff(merged.subarray(blockSize), reference);
  assert.ok(error < 1e-5, `naive convolution error ${error}`);
});

test('正弦 + 短冲激响应：分块不变且匹配大 FFT 参考实现', () => {
  const blockSize = 256;
  const input = makeSine(2000, 1200, SR, 0.6);
  const impulse = makeSine(400, 300, SR, 0.5);
  const { merged } = runStream(input, impulse, blockSize, [333, 17, 4096]);
  const error = assertMatchesLinearConvolution(merged, blockSize, input, impulse, 1e-5);
  assert.ok(error > 0);
});

test('长冲激响应（跨多个分区）结果正确，分区数随 IR 长度增长但有界', () => {
  const blockSize = 512;
  const input = makeNoise(8000, 88);
  const impulse = makeDecayingNoiseIR(6000, 31);
  const convolver = new MonoConvolver(impulse, { sampleRate: SR, blockSize });
  assert.equal(convolver.partitionCount, Math.ceil(impulse.length / blockSize));
  assert.equal(convolver.tailLength, blockSize + impulse.length - 1);

  const { merged } = runStream(input, impulse, blockSize, [123, 4096, 77], {});
  const reference = fftConvolution(input, impulse);
  const error = maxAbsDiff(merged.subarray(blockSize), reference);
  // 6000 抽头、8000 输入，累积误差仍保持在 1e-4 以内。
  assert.ok(error < 1e-4, `long IR error ${error}`);
});

test('flush 输出完整混响尾巴，且尾部能量随时间衰减到零附近', () => {
  const blockSize = 128;
  const input = makeNoise(1000, 3);
  const impulse = makeDecayingNoiseIR(500, 9);
  const { tail, merged } = runStream(input, impulse, blockSize, [256]);

  assert.equal(tail.length, blockSize + impulse.length - 1);
  // 尾巴末尾应接近静默（IR 已指数衰减）。
  const ending = tail.subarray(tail.length - 32);
  assert.ok(rms(ending) < 1e-3, `tail ending rms ${rms(ending)}`);

  // 尾巴拼回后与一次性线性卷积一致。
  const reference = fftConvolution(input, impulse);
  const error = maxAbsDiff(merged.subarray(blockSize), reference);
  assert.ok(error < 1e-4);
});

test('wet/dry 混合：纯 dry 等于延迟输入，混合结果等于两路加权和', () => {
  const blockSize = 64;
  const input = makeNoise(1500, 66);
  const impulse = makeDecayingNoiseIR(300, 14);

  const dryOnly = runStream(input, impulse, blockSize, [200], { wet: 0, dry: 1 });
  const dryAligned = dryOnly.merged.subarray(blockSize, blockSize + input.length);
  assert.ok(maxAbsDiff(dryAligned, input) < 1e-6);

  const wetOnly = runStream(input, impulse, blockSize, [200], { wet: 1, dry: 0 });
  const mixed = runStream(input, impulse, blockSize, [200], { wet: 0.3, dry: 0.7 });
  const tolerance = 1e-6;
  for (let i = 0; i < wetOnly.merged.length; i++) {
    const expected = 0.3 * wetOnly.merged[i] + 0.7 * dryOnly.merged[i];
    assert.ok(
      Math.abs(mixed.merged[i] - expected) < tolerance,
      `mix mismatch at ${i}: ${mixed.merged[i]} vs ${expected}`,
    );
  }
});

test('立体声左右声道独立处理，无串音', () => {
  const blockSize = 128;
  const irLeft = makeDecayingNoiseIR(400, 21);
  const irRight = makeDecayingNoiseIR(400, 22);
  const convolver = new StereoConvolver([irLeft, irRight], {
    sampleRate: SR,
    blockSize,
  });

  const leftInput = makeNoise(2000, 101);
  const rightInput = makeNoise(2000, 202);
  const leftOut = [];
  const rightOut = [];
  const chunks = [100, 333, 7, 512];
  let offset = 0;
  let plan = 0;
  while (offset < leftInput.length) {
    const size = Math.min(chunks[plan % chunks.length], leftInput.length - offset);
    const [lo, ro] = convolver.process([
      leftInput.subarray(offset, offset + size),
      rightInput.subarray(offset, offset + size),
    ]);
    assert.equal(lo.length, size);
    assert.equal(ro.length, size);
    leftOut.push(lo);
    rightOut.push(ro);
    offset += size;
    plan++;
  }
  const [tailL, tailR] = convolver.flush();

  const merge = (parts, tail) => {
    const total = parts.reduce((s, p) => s + p.length, 0) + tail.length;
    const out = new Float32Array(total);
    let pos = 0;
    for (const part of parts) {
      out.set(part, pos);
      pos += part.length;
    }
    out.set(tail, pos);
    return out;
  };

  const mergedL = merge(leftOut, tailL);
  const mergedR = merge(rightOut, tailR);
  const refL = fftConvolution(leftInput, irLeft);
  const refR = fftConvolution(rightInput, irRight);
  assert.ok(maxAbsDiff(mergedL.subarray(blockSize), refL) < 1e-4);
  assert.ok(maxAbsDiff(mergedR.subarray(blockSize), refR) < 1e-4);

  // 串音检查：右输入静音时，右输出必须全零，左输出不变。
  const iso = new StereoConvolver([irLeft, irRight], { sampleRate: SR, blockSize });
  const [isoL, isoR] = iso.process([leftInput.subarray(0, 500), new Float32Array(500)]);
  const [isoTailL, isoTailR] = iso.flush();
  for (const v of isoR) assert.equal(v, 0);
  for (const v of isoTailR) assert.equal(v, 0);
  const direct = new MonoConvolver(irLeft, { sampleRate: SR, blockSize });
  const directOut = direct.process(leftInput.subarray(0, 500));
  assert.ok(maxAbsDiff(isoL, directOut) === 0);
  assert.ok(isoTailL.length === direct.flush().length);
});

test('运行期内存不随输入时长增长', () => {
  const blockSize = 128;
  const impulse = makeDecayingNoiseIR(640, 4);
  const convolver = new MonoConvolver(impulse, { sampleRate: SR, blockSize });
  const input = makeNoise(20000, 5);

  const measure = () =>
    convolver._slots.reduce(
      (sum, slot) => sum + slot.real.length + slot.imag.length,
      0,
    ) + convolver._queue.length + convolver._pending.length;

  convolver.process(input.subarray(0, 5000));
  const afterFirst = measure();
  convolver.process(input.subarray(5000, 15000));
  const afterMore = measure();
  convolver.process(input.subarray(15000));
  const afterAll = measure();

  // 槽位数量固定，队列与待处理缓冲都小于一个块规模。
  assert.equal(convolver._slots.length, Math.ceil(impulse.length / blockSize));
  assert.ok(afterMore <= afterFirst + blockSize * 4);
  assert.ok(afterAll <= afterFirst + blockSize * 4);
  assert.ok(convolver._pending.length < blockSize);
  assert.ok(convolver._queue.length < blockSize * 2);
});

test('单分区（IR 短于一个块）与边界块大小均可工作', () => {
  const blockSize = 16;
  const input = makeSine(137, 200, SR);
  const impulse = makeDecayingNoiseIR(10, 2);
  const convolver = new MonoConvolver(impulse, { sampleRate: SR, blockSize });
  assert.equal(convolver.partitionCount, 1);
  const { merged } = runStream(input, impulse, blockSize, [1, 16, 13]);
  const reference = naiveConvolution(input, impulse);
  assert.ok(maxAbsDiff(merged.subarray(blockSize), reference) < 1e-5);
});
