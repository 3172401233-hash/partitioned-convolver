import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MonoConvolver, StereoConvolver } from '../lib/convolver.js';
import { makeNoise } from './helpers.js';

const SR = 48000;
const validIR = new Float32Array([0.5, 0.3, 0.1]);

test('空冲激响应直接报错', () => {
  assert.throws(
    () => new MonoConvolver(new Float32Array(0), { sampleRate: SR, blockSize: 64 }),
    /impulseResponse must not be empty/,
  );
});

test('分区大小不是 2 的幂直接报错', () => {
  for (const bad of [0, -64, 1000, 1.5, 96]) {
    assert.throws(
      () => new MonoConvolver(validIR, { sampleRate: SR, blockSize: bad }),
      /blockSize must be a power of two/,
    );
  }
});

test('冲激响应含非法浮点数直接报错', () => {
  const badValues = [NaN, Infinity, -Infinity];
  for (const bad of badValues) {
    const ir = new Float32Array([0.5, bad, 0.1]);
    assert.throws(
      () => new MonoConvolver(ir, { sampleRate: SR, blockSize: 64 }),
      /non-finite value/,
    );
  }
});

test('process 输入含非法浮点数直接报错', () => {
  const convolver = new MonoConvolver(validIR, { sampleRate: SR, blockSize: 64 });
  const badValues = [NaN, Infinity, -Infinity];
  for (const bad of badValues) {
    assert.throws(
      () => convolver.process(new Float32Array([0.1, bad, 0.2])),
      /non-finite value/,
    );
  }
});

test('空输入直接报错', () => {
  const convolver = new MonoConvolver(validIR, { sampleRate: SR, blockSize: 64 });
  assert.throws(() => convolver.process(new Float32Array(0)), /input must not be empty/);
});

test('采样率不一致直接报错', () => {
  const convolver = new MonoConvolver(validIR, { sampleRate: 48000, blockSize: 64 });
  assert.throws(
    () => convolver.process(makeNoise(100), 44100),
    /sampleRate mismatch/,
  );
});

test('非法采样率与非法 wet/dry 直接报错', () => {
  assert.throws(
    () => new MonoConvolver(validIR, { sampleRate: 0, blockSize: 64 }),
    /positive integer/,
  );
  assert.throws(
    () => new MonoConvolver(validIR, { sampleRate: SR, blockSize: 64, wet: NaN }),
    /wet must be a finite number/,
  );
  assert.throws(
    () => new MonoConvolver(validIR, { sampleRate: SR, blockSize: 64, dry: Infinity }),
    /dry must be a finite number/,
  );
});

test('flush 之后不能继续处理，重复 flush 也报错', () => {
  const convolver = new MonoConvolver(validIR, { sampleRate: SR, blockSize: 64 });
  convolver.process(makeNoise(100));
  convolver.flush();
  assert.throws(() => convolver.process(makeNoise(10)), /has been flushed/);
  assert.throws(() => convolver.flush(), /already been flushed/);
});

test('reset 后引擎可重新使用', () => {
  const convolver = new MonoConvolver(validIR, { sampleRate: SR, blockSize: 64 });
  convolver.process(makeNoise(100));
  convolver.flush();
  convolver.reset();
  const out = convolver.process(makeNoise(100));
  assert.equal(out.length, 100);
  const tail = convolver.flush();
  assert.equal(tail.length, 64 + validIR.length - 1);
});

test('立体声参数与声道长度校验', () => {
  assert.throws(
    () => new StereoConvolver([validIR], { sampleRate: SR, blockSize: 64 }),
    /two impulse responses/,
  );
  const convolver = new StereoConvolver([validIR, validIR], {
    sampleRate: SR,
    blockSize: 64,
  });
  assert.throws(
    () => convolver.process([makeNoise(10), makeNoise(11)]),
    /input lengths differ/,
  );
});

test('输入类型错误直接报错', () => {
  const convolver = new MonoConvolver(validIR, { sampleRate: SR, blockSize: 64 });
  assert.throws(() => convolver.process([0.1, 0.2]), /Float32Array/);
  assert.throws(
    () => new MonoConvolver([0.1, 0.2], { sampleRate: SR, blockSize: 64 }),
    /Float32Array/,
  );
});
