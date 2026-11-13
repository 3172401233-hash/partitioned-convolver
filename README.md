# partitioned-convolver

基于自研 FFT 的流式分区卷积（partitioned convolution）混响内核，面向游戏音频里较长脉冲响应
（impulse response, IR）的实时场景声处理。纯 JavaScript 实现，**无任何第三方依赖，不包含播放器**。

- 均匀分区 overlap-save（UPC），把逐样本 O(N·M) 卷积降到每块 O(P·N·log N)，N = 2B。
- 流式 `process()` 每次返回与输入**等长**的数据，内部状态跨调用保留，任意 buffer 大小
  送入都不会在接缝处产生差异（逐样本一致）。
- 结束时 `flush()` 单独输出混响尾巴。
- 运行期内存只与分区数 P 相关，**不随输入时长增长**。
- 单声道与立体声均可，立体声左右使用两个完全独立的引擎，无串音。
- 可调 `wet` / `dry` 混合比例，dry 路径按算法延迟对齐相位。

## 目录结构

```
lib/fft.js         自研迭代 radix-2 Cooley-Tukey FFT（复数，分离实部/虚部，原地运算）
lib/convolver.js   MonoConvolver / StereoConvolver 分区卷积引擎
test/              node:test 测试（FFT、卷积正确性、分块不变性、立体声、错误分支等）
```

## 算法概要

设分区长度为 `B`（即 `blockSize`，必须是 2 的幂），FFT 长度 `N = 2B`，IR 长度为 `L`，
分区数 `P = ceil(L / B)`。

1. 初始化时把 IR 切成 P 个长度 B 的分区，各自零填充到 N 并做一次 FFT，得到 `H[p]`。
2. 每个输入块与上一块拼成 overlap-save 缓冲 `[旧B, 新B]`，FFT 得 `X[n]`，写入长度为 P
   的环形频谱槽。
3. 频域累加 `Y[n] = Σ_p X[n-p] · H[p]`（复数乘加），逆 FFT 后取后半段 B 个有效样本。
4. 引擎预填 B 个零样本，因此存在固定 `B` 个样本（`latency`）的算法延迟，但保证
   `process()` 永远返回与本次输入等长的输出，任意分块节奏结果完全相同。
5. `flush()` 先把不足一块的残留输入补零成真实块，再喂零块（写入零频谱槽）排空所有分区，
   返回长度 `latency + L - 1` 的混响尾巴。

内存上界固定为 P 个复数频谱槽（各 2N 个 `Float64`），加上小于 2B 的输出队列与小于 B 的
待处理缓冲，与已经处理的音频总时长无关。FFT 内部全程使用 `Float64` 累加，输入输出为
`Float32Array`。

## 用法

```js
import { MonoConvolver, StereoConvolver } from './lib/convolver.js';

// 单声道
const convolver = new MonoConvolver(impulseResponse, {
  sampleRate: 48000, // 必填，正整数
  blockSize: 1024,   // 可选，默认 1024，必须是 2 的幂
  wet: 1,            // 可选，默认 1（纯湿声）
  dry: 0,            // 可选，默认 0
});

// input 是 Float32Array，任意长度；返回等长 Float32Array
const out = convolver.process(input);
// 输入结束后取尾巴，长度 = blockSize + irLength - 1
const tail = convolver.flush();

// 立体声：传入 [左IR, 右IR]，process 传入 [左输入, 右输入]
const stereo = new StereoConvolver([irLeft, irRight], { sampleRate: 48000, blockSize: 512 });
const [outL, outR] = stereo.process([inputL, inputR]);
const [tailL, tailR] = stereo.flush();
```

### 实例属性

| 属性 | 含义 |
| --- | --- |
| `latency` | 算法延迟，固定等于 `blockSize` 个样本 |
| `tailLength` | `flush()` 返回的尾巴长度，`blockSize + irLength - 1` |
| `partitionCount` | IR 分区数 P，运行期环形槽数量 |
| `blockSize` / `fftSize` | 分区长度 B 与 FFT 长度 2B |
| `wet` / `dry` | 湿声 / 干声增益 |
| `channels` | `1` 或 `2` |

`reset()` 清空全部运行期状态（环形槽、队列、延迟缓冲），可重新处理新音频。

## 输入校验与错误

下列情况直接抛出带清晰信息的 `RangeError` / `TypeError`，不产出脏数据：

- 空输入：`process()` 收到长度为 0 的 `Float32Array`。
- 非法浮点数：输入或 IR 含 `NaN` / `Infinity` / `-Infinity`（错误信息带样本下标）。
- 采样率不一致：`process()` 的采样率与构造引擎时不同。
- 脉冲响应为空：构造时 IR 长度为 0。
- 分区大小不是 2 的幂：`blockSize` 非法（含 0、负数、非整数、非幂值）。
- 非法 `wet` / `dry`（非有限数）、非法 `sampleRate`、输入不是 `Float32Array`。
- `flush()` 之后继续 `process()`、重复 `flush()`（需先 `reset()`）。
- 立体声参数数量不对、左右输入长度不一致、左右 IR 采样率不同。

## 测试与数值误差边界

运行：

```bash
npm test        # 等价于 node --test
```

测试覆盖（见 `test/`）：

- 自研 FFT：往返变换误差、Parseval 能量守恒、与朴素 DFT 逐项对比、单位脉冲/直流频谱。
- 单位脉冲 IR：输出等于延迟后的输入，结束后全零。
- 正弦与确定性噪声生成的输入，配合多种 IR（短 IR、指数衰减噪声 IR、跨多分区的长 IR）。
- **分块不变性**：同一段音频用 1、7、B、3B、斐波那契序列等 7 种 buffer 切分，输出
  逐样本完全一致（`maxAbsDiff === 0`），接缝无差异。
- `process()` 始终返回等长数据；结果对照朴素直接卷积与单次大 FFT 卷积两套独立参考实现。
- 长脉冲响应（6000 抽头、跨 12 个分区）正确性与有界分区数。
- `flush()` 尾巴长度正确、末端衰减到静默、拼回后等于完整线性卷积。
- 立体声无串音（一侧静音时该侧输出全零，另一侧与单声道引擎逐样本相同）。
- wet/dry：纯 dry 等于延迟输入，混合结果等于两路加权和。
- 内存有界：处理 1k / 10k / 50k / 100k 样本后环形槽数量恒定，队列始终小于 2B。
- 上述全部错误分支。

### 实测误差边界（Node v22，Float64 FFT + Float32 输入输出）

在本机执行 `node --test` 与误差测量脚本得到：

| 场景 | 最大绝对误差 | 最大相对误差 |
| --- | --- | --- |
| 短 IR（90 抽头）对照朴素卷积 | 1.4e-16 | 1.9e-16 |
| 正弦、B=256 对照大 FFT | 6.7e-15 | 2.8e-15 |
| 长 IR（6000 抽头，8000 输入） | 7.1e-15 | 3.0e-16 |
| 噪声、B=64 分块 | 0（逐样本一致） | 0 |
| 立体声左/右声道 | 0（逐样本一致） | 0 |

结论：相对线性卷积真值，**最大绝对误差不超过 1e-14、相对误差不超过 1e-14**（机器精度
量级）；同一段音频不同分块方式之间为**逐样本零误差**。测试内断言采用更宽松的保守上界
（`1e-4` ~ `1e-5`，按用例规模区分），为浮点实现差异留足余量。

## 最近一次测试结果

```
# tests 26
# pass 26
# fail 0
# duration_ms ~270
```
