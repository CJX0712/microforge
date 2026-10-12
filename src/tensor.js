/**
 * tensor.js — 极小 reverse-mode 自动微分引擎（Float32 张量 + tape）
 *
 * 血统（复用而非自造）：
 *   - 计算图/tape 反向模式：micrograd (Karpathy, 2020) 思想，升级到 N-D 张量 + 原地累加
 *   - 数值稳定：log-sum-exp 技巧（标准做法，见任何现代 softmax 实现）
 *   - 梯度检验：中心差分 (f(x+h)-f(x-h))/2h —— SQP/数值分析教科书金标准
 *
 * 设计约束：
 *   - 数据一律 Float32Array，扁平存储，行主序（C-order）
 *   - 每个算子只手写一条 backward，全部由 tests/invariants.js 的梯度检验钉死
 *   - 零依赖，Node 与浏览器同构
 *
 * @author 晨星
 */

export let TAPE = [];          // 计算图节点，backward 时逆序回放
export let NO_GRAD = false;    // 推理/生成时置 true，不建图，省内存

export function setNoGrad(v) { NO_GRAD = v; }

export class Tensor {
  /**
   * @param {Float32Array|number[]} data 扁平数据
   * @param {number[]} shape
   * @param {{requiresGrad?:boolean, name?:string}} [opts]
   */
  constructor(data, shape, opts = {}) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape.slice();
    this.size = this.data.length;
    this.grad = null;
    this.requiresGrad = !!opts.requiresGrad;
    this.name = opts.name || '';
    this._back = null;      // () => void，负责把 out.grad 累加回父节点
    this._parents = null;   // 仅用于可视化/调试
    if (this.requiresGrad && !NO_GRAD) {
      this.grad = new Float32Array(this.size);
      TAPE.push(this);
    }
  }

  static zeros(shape) {
    let n = 1; for (const s of shape) n *= s;
    return new Tensor(new Float32Array(n), shape);
  }
  static ones(shape) {
    const t = Tensor.zeros(shape); t.data.fill(1); return t;
  }
  /** 高斯初始化（Box-Muller），确定性 RNG 保证可复现 */
  static randn(shape, rng, std = 1) {
    let n = 1; for (const s of shape) n *= s;
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = std * gaussian(rng);
    return new Tensor(d, shape);
  }
  static uniform(shape, rng, lo, hi) {
    let n = 1; for (const s of shape) n *= s;
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = lo + (hi - lo) * rng();
    return new Tensor(d, shape);
  }

  /** 零化梯度（每步训练前调用） */
  zeroGrad() { if (this.grad) this.grad.fill(0); }

  /** 把一个普通张量升级为可训练参数（分配 grad，但不进计算图） */
  requireGrad() {
    this.requiresGrad = true;
    if (!this.grad) this.grad = new Float32Array(this.size);
    return this;
  }

  /** 从叶子节点反向传播 */
  backward() {
    if (!this.grad) this.grad = new Float32Array(this.size);
    this.grad.fill(0);
    this.grad[0] = 1;                       // dL/dL = 1
    for (let i = TAPE.length - 1; i >= 0; i--) {
      const node = TAPE[i];
      if (node._back) node._back();
    }
  }

  /** 全参数梯度 L2 范数（监控用） */
  static gradNorm(list) {
    let s = 0;
    for (const p of list) if (p.grad) for (let i = 0; i < p.grad.length; i++) s += p.grad[i] * p.grad[i];
    return Math.sqrt(s);
  }
  static paramNorm(list) {
    let s = 0;
    for (const p of list) for (let i = 0; i < p.data.length; i++) s += p.data[i] * p.data[i];
    return Math.sqrt(s);
  }
}

/** 确定性 PRNG：mulberry32 —— 32 位、快、可复现，跨 Node/浏览器一致 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller：u1,u2 ~ U(0,1) -> z ~ N(0,1) */
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function clearTape() { TAPE = []; }

/* ------------------------------------------------------------------ *
 *  算子集合（每个算子：forward + 对应 backward）
 * ------------------------------------------------------------------ */

/** C = A @ B，支持 2D [M,K]x[K,N] 与 3D batched [B,M,K]x[B,K,N] */
export function matmul(a, b) {
  const rank = a.shape.length;
  if (rank === 2) {
    const [M, K] = a.shape, K2 = b.shape[0], N = b.shape[1];
    if (K !== K2) throw new Error(`matmul shape mismatch: ${a.shape} @ ${b.shape}`);
    const out = new Float32Array(M * N);
    const A = a.data, B = b.data;
    // i-k-j 循环顺序：内层连续访问 B 行与 out 行，缓存友好
    for (let i = 0; i < M; i++) {
      const oi = i * N;
      for (let k = 0; k < K; k++) {
        const aik = A[i * K + k];
        if (aik === 0) continue;
        const bk = k * N;
        for (let j = 0; j < N; j++) out[oi + j] += aik * B[bk + j];
      }
    }
    const o = new Tensor(out, [M, N]);
    if (a.requiresGrad || b.requiresGrad) {
      o.requireGrad(); TAPE.push(o);
      o._back = () => {
        const G = o.grad;
        if (a.requiresGrad) {
          // dA = G @ B^T
          for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += G[i * N + j] * B[k * N + j];
            a.grad[i * K + k] += s;
          }
        }
        if (b.requiresGrad) {
          // dB = A^T @ G
          for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
            let s = 0;
            for (let i = 0; i < M; i++) s += A[i * K + k] * G[i * N + j];
            b.grad[k * N + j] += s;
          }
        }
      };
    }
    return o;
  }
  if (rank === 3) {
    const [B_, M, K] = a.shape, [B2, K2, N] = b.shape;
    if (B_ !== B2 || K !== K2) throw new Error(`bmm shape mismatch: ${a.shape} @ ${b.shape}`);
    const out = new Float32Array(B_ * M * N);
    for (let b0 = 0; b0 < B_; b0++) {
      const ao = b0 * M * K, bo = b0 * K * N, oo = b0 * M * N;
      for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
        const aik = a.data[ao + i * K + k];
        if (aik === 0) continue;
        for (let j = 0; j < N; j++) out[oo + i * N + j] += aik * b.data[bo + k * N + j];
      }
    }
    const o = new Tensor(out, [B_, M, N]);
    if (a.requiresGrad || b.requiresGrad) {
      o.requireGrad(); TAPE.push(o);
      o._back = () => {
        const G = o.grad;
        for (let b0 = 0; b0 < B_; b0++) {
          const ao = b0 * M * K, bo = b0 * K * N, oo = b0 * M * N;
          if (a.requiresGrad) for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += G[oo + i * N + j] * b.data[bo + k * N + j];
            a.grad[ao + i * K + k] += s;
          }
          if (b.requiresGrad) for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
            let s = 0;
            for (let i = 0; i < M; i++) s += a.data[ao + i * K + k] * G[oo + i * N + j];
            b.grad[bo + k * N + j] += s;
          }
        }
      };
    }
    return o;
  }
  throw new Error('matmul: only 2D / 3D-batched supported');
}

/**
 * 线性层：out[T,Dout] = x[T,Din] @ W[Dout,Din]ᵀ
 * 与 PyTorch 的 F.linear / nanoGPT 的语义一致（权重存 [out,in]，前向不物化转置）
 * 血统：约定来自 PyTorch nn.Linear
 */
export function linear(x, W) {
  const [T, Din] = x.shape, [Dout, Din2] = W.shape;
  if (Din !== Din2) throw new Error(`linear shape mismatch: ${x.shape} @ W${W.shape}`);
  const out = new Float32Array(T * Dout);
  const X = x.data, w = W.data;
  for (let i = 0; i < T; i++) {
    const oi = i * Dout, xi = i * Din;
    for (let j = 0; j < Dout; j++) {
      let s = 0; const wj = j * Din;
      for (let k = 0; k < Din; k++) s += X[xi + k] * w[wj + k];
      out[oi + j] = s;
    }
  }
  const o = new Tensor(out, [T, Dout]);
  if (x.requiresGrad || W.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      const G = o.grad;
      if (x.requiresGrad) for (let i = 0; i < T; i++) {
        const gi = i * Dout, xi = i * Din;
        for (let k = 0; k < Din; k++) {
          let s = 0;
          for (let j = 0; j < Dout; j++) s += G[gi + j] * w[j * Din + k];
          x.grad[xi + k] += s;
        }
      }
      if (W.requiresGrad) for (let j = 0; j < Dout; j++) {
        const wj = j * Din;
        for (let k = 0; k < Din; k++) {
          let s = 0;
          for (let i = 0; i < T; i++) s += G[i * Dout + j] * X[i * Din + k];
          W.grad[wj + k] += s;
        }
      }
    };
  }
  return o;
}

/** 逐元素相加，支持 (2D, 1D) 行广播：out[i,j] = a[i,j] + b[j] */
export function add(a, b) {
  const broadcast = a.shape.length === 2 && b.shape.length === 1;
  const out = new Float32Array(a.size);
  if (broadcast) {
    const [M, N] = a.shape;
    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) out[i * N + j] = a.data[i * N + j] + b.data[j];
  } else {
    for (let i = 0; i < a.size; i++) out[i] = a.data[i] + b.data[i];
  }
  const o = new Tensor(out, a.shape);
  if (a.requiresGrad || b.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += o.grad[i];
      if (b.requiresGrad) {
        if (broadcast) {
          const [M, N] = a.shape;
          for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) b.grad[j] += o.grad[i * N + j];
        } else for (let i = 0; i < b.size; i++) b.grad[i] += o.grad[i];
      }
    };
  }
  return o;
}

/** 逐元素相乘 */
export function mul(a, b) {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] * b.data[i];
  const o = new Tensor(out, a.shape);
  if (a.requiresGrad || b.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += o.grad[i] * b.data[i];
      if (b.requiresGrad) for (let i = 0; i < b.size; i++) b.grad[i] += o.grad[i] * a.data[i];
    };
  }
  return o;
}

/** 标量缩放 */
export function scale(a, s) {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] * s;
  const o = new Tensor(out, a.shape);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => { for (let i = 0; i < a.size; i++) a.grad[i] += o.grad[i] * s; };
  }
  return o;
}

/** SiLU / Swish: x * sigmoid(x) —— 反向 1 行闭式解 */
export function silu(a) {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) {
    const x = a.data[i];
    out[i] = x / (1 + Math.exp(-x));
  }
  const o = new Tensor(out, a.shape);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let i = 0; i < a.size; i++) {
        const x = a.data[i], s = 1 / (1 + Math.exp(-x));
        // d/dx [x*s] = s + x*s*(1-s)
        a.grad[i] += o.grad[i] * (s + x * s * (1 - s));
      }
    };
  }
  return o;
}

/**
 * 行 softmax，数值稳定：先减行最大值。
 * 不变量：softmax(x + c) == softmax(x)（平移不变），由 tests 钉死。
 */
export function softmaxRows(a) {
  const [M, N] = a.shape;
  const out = new Float32Array(a.size);
  for (let i = 0; i < M; i++) {
    const oi = i * N;
    let mx = -Infinity;
    for (let j = 0; j < N; j++) if (a.data[oi + j] > mx) mx = a.data[oi + j];
    let sum = 0;
    for (let j = 0; j < N; j++) { const e = Math.exp(a.data[oi + j] - mx); out[oi + j] = e; sum += e; }
    const inv = 1 / sum;
    for (let j = 0; j < N; j++) out[oi + j] *= inv;
  }
  const o = new Tensor(out, a.shape);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let i = 0; i < M; i++) {
        const oi = i * N;
        let dot = 0;
        for (let j = 0; j < N; j++) dot += o.grad[oi + j] * out[oi + j];
        for (let j = 0; j < N; j++) a.grad[oi + j] += out[oi + j] * (o.grad[oi + j] - dot);
      }
    };
  }
  return o;
}

/**
 * RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * w
 * 反向（解析推导，见 README 附录 A）：
 *   g = dy * w ;  dx = (1/r) * (g - xhat * mean(g * xhat)) ; dw = dy * xhat
 * 血统：Zhang & Sennrich 2019（RMSNorm），已取代 LayerNorm 成为 LLM 事实标准
 */
export function rmsnorm(a, w, eps = 1e-6) {
  // 支持 2D [M,D] 与 3D [B,T,D]：一律沿最后一维归一化
  const D = a.shape[a.shape.length - 1];
  const M = a.size / D;
  if (w.size !== D) throw new Error(`rmsnorm: weight dim ${w.size} != ${D}`);
  const out = new Float32Array(a.size);
  const inv = new Float32Array(M);     // 1/r
  const xhat = new Float32Array(a.size);
  for (let i = 0; i < M; i++) {
    const oi = i * D;
    let ss = 0;
    for (let j = 0; j < D; j++) ss += a.data[oi + j] * a.data[oi + j];
    const r = Math.sqrt(ss / D + eps);
    inv[i] = 1 / r;
    for (let j = 0; j < D; j++) {
      const xh = a.data[oi + j] * inv[i];
      xhat[oi + j] = xh;
      out[oi + j] = xh * w.data[j];
    }
  }
  const o = new Tensor(out, a.shape);
  if (a.requiresGrad || w.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let i = 0; i < M; i++) {
        const oi = i * D;
        let meanGX = 0;
        for (let j = 0; j < D; j++) {
          const g = o.grad[oi + j] * w.data[j];
          meanGX += g * xhat[oi + j];
        }
        meanGX /= D;
        if (a.requiresGrad) for (let j = 0; j < D; j++) {
          const g = o.grad[oi + j] * w.data[j];
          a.grad[oi + j] += inv[i] * (g - xhat[oi + j] * meanGX);
        }
        if (w.requiresGrad) for (let j = 0; j < D; j++) w.grad[j] += o.grad[oi + j] * xhat[oi + j];
      }
    };
  }
  return o;
}

/**
 * RoPE 旋转位置编码（interleaved pair 形式，RoFormer 原文）
 *   q'_{2k}   = q_{2k}   cos(mθ_k) − q_{2k+1} sin(mθ_k)
 *   q'_{2k+1} = q_{2k}   sin(mθ_k) + q_{2k+1} cos(mθ_k)
 *   θ_k = 1 / base^(2k/D)
 * 血统：Su et al. 2021 (RoFormer)；现为 LLaMA/Mistral/Qwen 全家桶标配
 * 只支持 3D [B,T,D] 输入，D 为偶数
 */
export function rope(a, base = 10000, posOffset = 0) {
  // 支持 3D [B,T,D] 与 4D [B,H,T,dh]（4D 时把 B,H 视作合并 batch）
  const is4 = a.shape.length === 4;
  const B = is4 ? a.shape[0] * a.shape[1] : a.shape[0];
  const T = is4 ? a.shape[2] : a.shape[1];
  const D = is4 ? a.shape[3] : a.shape[2];
  if (D % 2 !== 0) throw new Error('rope: head dim must be even');
  const out = new Float32Array(a.size);
  const cosT = new Float32Array(T * (D / 2));
  const sinT = new Float32Array(T * (D / 2));
  for (let t = 0; t < T; t++) {
    const m = t + posOffset;
    for (let k = 0; k < D / 2; k++) {
      const theta = m / Math.pow(base, (2 * k) / D);
      cosT[t * (D / 2) + k] = Math.cos(theta);
      sinT[t * (D / 2) + k] = Math.sin(theta);
    }
  }
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
    const oi = (b * T + t) * D;
    for (let k = 0; k < D / 2; k++) {
      const c = cosT[t * (D / 2) + k], s = sinT[t * (D / 2) + k];
      const x0 = a.data[oi + 2 * k], x1 = a.data[oi + 2 * k + 1];
      out[oi + 2 * k] = x0 * c - x1 * s;
      out[oi + 2 * k + 1] = x0 * s + x1 * c;
    }
  }
  const o = new Tensor(out, a.shape);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    // R 是分块正交矩阵 ⇒ 反向即 R^T（旋转 −θ），无需存储中间量
    o._back = () => {
      for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
        const oi = (b * T + t) * D;
        for (let k = 0; k < D / 2; k++) {
          const c = cosT[t * (D / 2) + k], s = sinT[t * (D / 2) + k];
          const g0 = o.grad[oi + 2 * k], g1 = o.grad[oi + 2 * k + 1];
          a.grad[oi + 2 * k] += g0 * c + g1 * s;
          a.grad[oi + 2 * k + 1] += -g0 * s + g1 * c;
        }
      }
    };
  }
  return o;
}

/** Embedding 查表：W[V,D]，idx[T] -> [T,D]（反向为 scatter-add） */
export function embedding(W, idx) {
  const [V, D] = W.shape;
  const T = idx.length;
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const id = idx[t];
    if (id < 0 || id >= V) throw new Error(`embedding: id ${id} out of range [0,${V})`);
    out.set(W.data.subarray(id * D, id * D + D), t * D);
  }
  const o = new Tensor(out, [T, D]);
  if (W.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let t = 0; t < T; t++) {
        const g = t * D, p = idx[t] * D;
        for (let j = 0; j < D; j++) W.grad[p + j] += o.grad[g + j];
      }
    };
  }
  return o;
}

/**
 * 交叉熵：logits [T,V] vs targets Int32Array/Array [T] → 标量
 * 内部用 logsumexp 保证数值稳定（血统：标准做法）。
 * 反向闭式：dlogits = (softmax(logits) − onehot) / T
 */
export function crossEntropy(logits, targets, opts = {}) {
  const [T, V] = logits.shape;
  const sm = new Float32Array(T * V);
  let loss = 0;
  for (let t = 0; t < T; t++) {
    const oi = t * V;
    let mx = -Infinity;
    for (let j = 0; j < V; j++) if (logits.data[oi + j] > mx) mx = logits.data[oi + j];
    let sum = 0;
    for (let j = 0; j < V; j++) { const e = Math.exp(logits.data[oi + j] - mx); sm[oi + j] = e; sum += e; }
    const lse = mx + Math.log(sum);
    const inv = 1 / sum;
    for (let j = 0; j < V; j++) sm[oi + j] *= inv;
    loss += lse - logits.data[oi + targets[t]];
  }
  loss /= T;
  const o = new Tensor(new Float32Array([loss]), [1]);
  if (logits.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      const g = o.grad[0] / T;
      for (let t = 0; t < T; t++) {
        const oi = t * V;
        for (let j = 0; j < V; j++) logits.grad[oi + j] += g * (sm[oi + j] - (j === targets[t] ? 1 : 0));
      }
    };
  }
  o._probs = sm;   // 供推理/采样复用
  return o;
}

/** 转置 2D：[M,N] -> [N,M] */
export function transpose(a) {
  const [M, N] = a.shape;
  const out = new Float32Array(a.size);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) out[j * M + i] = a.data[i * N + j];
  const o = new Tensor(out, [N, M]);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) a.grad[i * N + j] += o.grad[j * M + i];
    };
  }
  return o;
}

/** reshape（仅形状改写，数据拷贝） */
export function reshape(a, shape) {
  let n = 1; for (const s of shape) n *= s;
  if (n !== a.size) throw new Error(`reshape: ${a.size} != ${n}`);
  const o = new Tensor(a.data.slice(), shape);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => { for (let i = 0; i < a.size; i++) a.grad[i] += o.grad[i]; };
  }
  return o;
}

/** 沿指定轴切分：split(a, sizes, axis) —— 用于 MHA 的头切分 */
export function splitHeads(a, nHeads) {
  // [B,T,D] -> [B,nHeads,T,dh]
  const [B, T, D] = a.shape;
  const dh = D / nHeads;
  const out = new Float32Array(a.size);
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < nHeads; h++)
    for (let d = 0; d < dh; d++)
      out[((b * nHeads + h) * T + t) * dh + d] = a.data[(b * T + t) * D + h * dh + d];
  const o = new Tensor(out, [B, nHeads, T, dh]);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < nHeads; h++)
        for (let d = 0; d < dh; d++)
          a.grad[(b * T + t) * D + h * dh + d] += o.grad[((b * nHeads + h) * T + t) * dh + d];
    };
  }
  return o;
}

/** mergeHeads: [B,nHeads,T,dh] -> [B,T,D]（splitHeads 的逆） */
export function mergeHeads(a) {
  const [B, H, T, dh] = a.shape;
  const out = new Float32Array(B * T * H * dh);
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < H; h++)
    for (let d = 0; d < dh; d++)
      out[(b * T + t) * (H * dh) + h * dh + d] = a.data[((b * H + h) * T + t) * dh + d];
  const o = new Tensor(out, [B, T, H * dh]);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < H; h++)
        for (let d = 0; d < dh; d++)
          a.grad[((b * H + h) * T + t) * dh + d] += o.grad[(b * T + t) * (H * dh) + h * dh + d];
    };
  }
  return o;
}

/** 按最后一维切片并拼接：[B,T,D] 取 [:, :, start:start+len] → 新 [B,T,len] */
export function sliceLast(a, start, len) {
  const [B, T, D] = a.shape;
  const out = new Float32Array(B * T * len);
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++)
    for (let j = 0; j < len; j++) out[(b * T + t) * len + j] = a.data[(b * T + t) * D + start + j];
  const o = new Tensor(out, [B, T, len]);
  if (a.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let b = 0; b < B; b++) for (let t = 0; t < T; t++)
        for (let j = 0; j < len; j++) a.grad[(b * T + t) * D + start + j] += o.grad[(b * T + t) * len + j];
    };
  }
  return o;
}

/** 就地缩放全部元素（不建图，供优化器使用） */
export function inplaceScale(t, s) { for (let i = 0; i < t.size; i++) t.data[i] *= s; }

export { gaussian };
