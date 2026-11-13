import { FFT, isPowerOfTwo } from './fft.js';

const MAX_PARTITIONS = 8192;

function assertFiniteArray(values, label) {
  if (!(values instanceof Float32Array)) {
    throw new TypeError(`${label} must be a Float32Array`);
  }
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new RangeError(`${label} contains a non-finite value at index ${i}`);
    }
  }
}

function assertSampleRate(rate) {
  if (!Number.isInteger(rate) || rate <= 0) {
    throw new RangeError(`sampleRate must be a positive integer, got ${rate}`);
  }
}

function assertGain(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number, got ${value}`);
  }
}

// 单声道均匀分区卷积（uniformly partitioned overlap-save）。
// 分区长度 B（样本），FFT 长度 N = 2B，冲激响应被切成 P 个分区，
// 运行期状态只与 P 相关，不随流式输入时长增长。
export class MonoConvolver {
  constructor(impulseResponse, options = {}) {
    assertFiniteArray(impulseResponse, 'impulseResponse');
    if (impulseResponse.length === 0) {
      throw new RangeError('impulseResponse must not be empty');
    }

    const sampleRate = options.sampleRate;
    assertSampleRate(sampleRate);

    const blockSize = options.blockSize ?? 1024;
    if (!Number.isInteger(blockSize) || !isPowerOfTwo(blockSize)) {
      throw new RangeError(`blockSize must be a power of two, got ${blockSize}`);
    }

    const wet = options.wet ?? 1;
    const dry = options.dry ?? 0;
    assertGain(wet, 'wet');
    assertGain(dry, 'dry');

    this.sampleRate = sampleRate;
    this.channels = 1;
    this.blockSize = blockSize;
    this.fftSize = blockSize * 2;
    this.irLength = impulseResponse.length;
    this.partitionCount = Math.ceil(this.irLength / blockSize);
    if (this.partitionCount > MAX_PARTITIONS) {
      throw new RangeError(
        `too many partitions: ${this.partitionCount} > ${MAX_PARTITIONS}`,
      );
    }

    this.wet = wet;
    this.dry = dry;
    // 算法引入 B 个样本的输出延迟，dry 路径必须对齐到同一相位。
    this.latency = blockSize;
    // 输入结束后还能输出的混响尾巴样本数。
    this.tailLength = this.latency + this.irLength - 1;

    this._fft = new FFT(this.fftSize);
    this._partitions = new Array(this.partitionCount);
    this._initPartitions(impulseResponse);
    this.reset();
  }

  _initPartitions(impulseResponse) {
    const n = this.fftSize;
    const b = this.blockSize;
    for (let p = 0; p < this.partitionCount; p++) {
      const real = new Float64Array(n);
      const imag = new Float64Array(n);
      const start = p * b;
      const end = Math.min(start + b, this.irLength);
      for (let i = start; i < end; i++) {
        // overlap-save：分区 IR 放在 FFT 缓冲前半段，后半段补零。
        real[i - start] = impulseResponse[i];
      }
      this._fft.transform(real, imag);
      this._partitions[p] = { real, imag };
    }
  }

  reset() {
    const n = this.fftSize;
    const b = this.blockSize;
    const slots = this.partitionCount;
    this._slots = new Array(slots);
    for (let i = 0; i < slots; i++) {
      this._slots[i] = { real: new Float64Array(n), imag: new Float64Array(n) };
    }
    this._blockCount = 0;
    // 最旧输入块，用于拼出 overlap-save 的 [旧 B, 新 B]。
    this._previousBlock = new Float32Array(b);
    this._pending = new Float32Array(0);
    // 预填 B 个 0，使每次 process 都能返回与输入等长的数据。
    this._queue = new Float32Array(b);
    this._finished = false;
  }

  // 处理一个完整的 B 样本块（块数据由 _pending 前部提供）。
  _processBlock(block) {
    const n = this.fftSize;
    const b = this.blockSize;

    // 保存当前块作为下一次的旧块（block 可能是 _pending 的视图，必须复制）。
    const currentBlock = Float32Array.from(block);

    const inputReal = new Float64Array(n);
    const inputImag = new Float64Array(n);
    for (let i = 0; i < b; i++) {
      // overlap-save 输入块：前半段为上一块 B 个样本，后半段为当前块。
      inputReal[i] = this._previousBlock[i];
      inputReal[i + b] = block[i];
    }
    this._fft.transform(inputReal, inputImag);

    // 先写槽再累加：保证 n>=P 时环形槽里的 X[n-p] 是有效历史输入。
    // flush 的零块写入零频谱槽，后续配对时天然贡献零，环形语义保持统一。
    const writeIndex = this._blockCount % this.partitionCount;
    const slot = this._slots[writeIndex];
    slot.real.set(inputReal);
    slot.imag.set(inputImag);

    const accReal = new Float64Array(n);
    const accImag = new Float64Array(n);
    for (let p = 0; p < this.partitionCount; p++) {
      const slotIndex = (this._blockCount - p + this.partitionCount) % this.partitionCount;
      const source = this._slots[slotIndex];
      const filter = this._partitions[p];
      for (let k = 0; k < n; k++) {
        const sr = source.real[k];
        const si = source.imag[k];
        const hr = filter.real[k];
        const hi = filter.imag[k];
        accReal[k] += sr * hr - si * hi;
        accImag[k] += sr * hi + si * hr;
      }
    }

    this._fft.inverse(accReal, accImag);

    this._previousBlock = currentBlock;
    this._blockCount++;

    const wet = this.wet;
    const dry = this.dry;
    const output = new Float32Array(b);
    if (wet === 1 && dry === 0) {
      for (let i = 0; i < b; i++) {
        output[i] = accReal[i + b];
      }
    } else {
      // dry 取当前块：该输出块延迟 B 个样本后才返回，正好与 wet 相位对齐。
      for (let i = 0; i < b; i++) {
        output[i] = wet * accReal[i + b] + dry * currentBlock[i];
      }
    }

    const merged = new Float32Array(this._queue.length + b);
    merged.set(this._queue, 0);
    merged.set(output, this._queue.length);
    this._queue = merged;
  }

  _drain(count) {
    const out = this._queue.slice(0, count);
    this._queue = this._queue.slice(count);
    return out;
  }

  // 任意长度流式输入，返回等长输出；内部按 B 样本分块，状态跨调用保留。
  process(input, inputSampleRate = this.sampleRate) {
    if (this._finished) {
      throw new Error('convolver has been flushed; call reset() before processing again');
    }
    assertFiniteArray(input, 'input');
    if (inputSampleRate !== this.sampleRate) {
      throw new RangeError(
        `sampleRate mismatch: convolver is ${this.sampleRate}, input is ${inputSampleRate}`,
      );
    }

    if (input.length === 0) {
      throw new RangeError('input must not be empty');
    }

    if (this._pending.length > 0) {
      const combined = new Float32Array(this._pending.length + input.length);
      combined.set(this._pending, 0);
      combined.set(input, this._pending.length);
      this._pending = combined;
    } else {
      this._pending = input.slice();
    }

    const b = this.blockSize;
    while (this._pending.length >= b) {
      this._processBlock(this._pending.subarray(0, b));
      this._pending = this._pending.slice(b);
    }

    return this._drain(input.length);
  }

  // 输入结束后冲洗混响尾巴，返回 latency + irLength - 1 个样本。
  flush() {
    if (this._finished) {
      throw new Error('convolver has already been flushed');
    }
    this._finished = true;

    const b = this.blockSize;
    // 先把不足一个块的残留输入补零成完整块，作为真实块参与卷积。
    if (this._pending.length > 0) {
      const padded = new Float32Array(b);
      padded.set(this._pending, 0);
      this._pending = new Float32Array(0);
      this._processBlock(padded);
    }
    // 再送零输入块排空全部分区；零块写入零频谱槽，不影响历史配对。
    const zeros = new Float32Array(b);
    while (this._queue.length < this.tailLength) {
      this._processBlock(zeros);
    }
    const tail = this._drain(this.tailLength);
    return tail;
  }
}

// 立体声包装器：两个完全独立的单声道引擎，左右声道互不串音。
export class StereoConvolver {
  constructor(impulseResponses, options = {}) {
    if (!Array.isArray(impulseResponses) || impulseResponses.length !== 2) {
      throw new RangeError('stereo convolver needs an array of two impulse responses [left, right]');
    }
    this._left = new MonoConvolver(impulseResponses[0], options);
    this._right = new MonoConvolver(impulseResponses[1], options);
    if (this._right.sampleRate !== this._left.sampleRate) {
      throw new RangeError('left and right impulse response sample rates differ');
    }
    this.sampleRate = this._left.sampleRate;
    this.channels = 2;
    this.blockSize = this._left.blockSize;
    this.fftSize = this._left.fftSize;
    this.irLength = Math.max(this._left.irLength, this._right.irLength);
    this.latency = this._left.latency;
    this.tailLength = Math.max(this._left.tailLength, this._right.tailLength);
    this.leftTailLength = this._left.tailLength;
    this.rightTailLength = this._right.tailLength;
    this.wet = this._left.wet;
    this.dry = this._left.dry;
  }

  process(channels, inputSampleRate = this.sampleRate) {
    if (!Array.isArray(channels) || channels.length !== 2) {
      throw new RangeError('stereo process() needs an array [leftInput, rightInput]');
    }
    const left = channels[0];
    const right = channels[1];
    if (left.length !== right.length) {
      throw new RangeError('left and right input lengths differ');
    }
    return [
      this._left.process(left, inputSampleRate),
      this._right.process(right, inputSampleRate),
    ];
  }

  flush() {
    return [this._left.flush(), this._right.flush()];
  }

  reset() {
    this._left.reset();
    this._right.reset();
  }
}
