// 自研迭代式 radix-2 Cooley-Tukey FFT，无第三方依赖。
// 复数以分离的实部 / 虚部 Float64Array 表示，原地运算。

function isPowerOfTwo(value) {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

export class FFT {
  constructor(size) {
    if (!Number.isInteger(size) || size <= 0 || !isPowerOfTwo(size)) {
      throw new RangeError(`FFT size must be a positive power of two, got ${size}`);
    }
    this.size = size;
    this.log2 = Math.log2(size);

    // 位反转置换表。
    this.bitReversal = new Int32Array(size);
    for (let i = 0; i < size; i++) {
      let reversed = 0;
      let x = i;
      for (let bit = 0; bit < this.log2; bit++) {
        reversed = (reversed << 1) | (x & 1);
        x >>>= 1;
      }
      this.bitReversal[i] = reversed;
    }

    // 每一级的旋转因子表，tables[k] 对应跨度 2^k 的蝶形运算。
    this.cosTables = new Array(this.log2);
    this.sinTables = new Array(this.log2);
    for (let k = 0; k < this.log2; k++) {
      const half = 1 << k;
      const span = half << 1;
      const cos = new Float64Array(half);
      const sin = new Float64Array(half);
      for (let j = 0; j < half; j++) {
        const angle = (-2 * Math.PI * j) / span;
        cos[j] = Math.cos(angle);
        sin[j] = Math.sin(angle);
      }
      this.cosTables[k] = cos;
      this.sinTables[k] = sin;
    }
  }

  // 原地正向 FFT，real / imag 长度均为 size。
  transform(real, imag) {
    const n = this.size;
    const rev = this.bitReversal;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = real[i];
        real[i] = real[j];
        real[j] = tr;
        const ti = imag[i];
        imag[i] = imag[j];
        imag[j] = ti;
      }
    }

    for (let k = 0; k < this.log2; k++) {
      const half = 1 << k;
      const span = half << 1;
      const cosTable = this.cosTables[k];
      const sinTable = this.sinTables[k];
      for (let group = 0; group < n; group += span) {
        for (let j = 0; j < half; j++) {
          const even = group + j;
          const odd = even + half;
          const wr = cosTable[j];
          const wi = sinTable[j];
          const ur = real[odd];
          const ui = imag[odd];
          const vr = ur * wr - ui * wi;
          const vi = ur * wi + ui * wr;
          const er = real[even];
          const ei = imag[even];
          real[odd] = er - vr;
          imag[odd] = ei - vi;
          real[even] = er + vr;
          imag[even] = ei + vi;
        }
      }
    }
  }

  // 原地逆 FFT，输出按 1/n 归一化。
  inverse(real, imag) {
    const n = this.size;
    for (let i = 0; i < n; i++) {
      imag[i] = -imag[i];
    }
    this.transform(real, imag);
    const inv = 1 / n;
    for (let i = 0; i < n; i++) {
      real[i] *= inv;
      imag[i] = -imag[i] * inv;
    }
  }
}

export { isPowerOfTwo };
