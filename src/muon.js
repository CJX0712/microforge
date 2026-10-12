/**
 * muon.js — Muon（Newton-Schulz 正交化动量） + AdamW 混合优化器
 *
 * 血统（全部复用世界顶级开源成果，非自造）：
 *   - Muon: Keller Jordan (2024) https://github.com/KellerJordan/Muon
 *           理论基础：Modular Duality in Deep Learning, arXiv:2410.21265
 *   - Newton-Schulz quintic 系数 (a,b,c) = (3.4445, −4.7750, 2.0315)：
 *           Jordan 的 polar express 修正版，5 步即可逼近矩阵符号函数 msign(M)=UVᵀ
 *   - Moonlight 缩放因子 0.2·√max(dout,din)：Moonshot AI (2025), arXiv:2502.16982
 *           使 Muon 的更新 RMS 与 AdamW 对齐 ⇒ 可直接复用 Adam 的学习率
 *   - AdamW: Loshchilov & Hutter 2019 (ICLR) —— 解耦权重衰减
 *
 * 分工（Moonlight 论文 3.5 节的工程共识）：
 *   2D 隐层矩阵（Wq,Wk,Wv,Wo,Wgate,Wup,Wdown）→ Muon
 *   1D 参数（RMSNorm 权重、bias）与嵌入/输出头       → AdamW
 *
 * @author 晨星
 */

/** 稠密矩阵乘：A[m,k] @ B[k,n] → C[m,n]（Float64 中间，避免 NS 迭代累积误差） */
function denseMul(A, B, m, k, n) {
  const C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const oi = i * n;
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p];
      if (a === 0) continue;
      const bp = p * n;
      for (let j = 0; j < n; j++) C[oi + j] += a * B[bp + j];
    }
  }
  return C;
}

/** 转置 [m,n] → [n,m] */
function denseT(X, m, n) {
  const T = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j * m + i] = X[i * n + j];
  return T;
}

/** 两套 Newton-Schulz 系数 */
export const NS_COEFFS = {
  /** Polar Express / Keller Jordan 五阶：工业界 Muon 实际使用，5 步即可，快 */
  quintic: { a: 3.4445, b: -4.7750, c: 2.0315 },
  /** 经典立方迭代 X ← (3X − X Xᵀ X)/2：数学上可证明收敛到真正半正交，用于交叉验证 */
  cubic: { a: 1.5, b: -0.5, c: 0 },
};

/**
 * Newton-Schulz 迭代：逼近 msign(G) = U Vᵀ（把全部奇异值压缩到 1 附近）
 *
 *   X ← G / ‖G‖_F
 *   repeat: A = X Xᵀ ; X ← a·X + (b·A + c·A²)·X
 *
 * 奇异值层面的迭代为 σ ← a·σ + b·σ³ + c·σ⁵：
 *   · quintic：一步就把 σ∈(0,1] 推到 ≈1 的邻域，5 步足够，故为训练默认
 *   · cubic  ：σ ← 1.5σ − 0.5σ³，不动点为 σ=1 且可证明收敛 ⇒ 用于不变量 ⑤ 的严格交叉验证
 *
 * 不变量（tests/invariants.js 钉死）：
 *   cubic  + 20 步 ⇒ ‖Gram(X) − I‖ < 1e-3（真正的半正交）
 *   quintic + 5 步 ⇒ 全部奇异值落在 [0.5, 1.5]（Polar Express 的经验包络）
 *
 * @param {Float64Array|Float32Array} G 扁平 [m,n]
 * @param {number} m @param {number} n @param {number} steps
 * @param {'quintic'|'cubic'} coeffs
 * @returns {Float64Array} 正交化后的 [m,n]
 */
export function newtonSchulz(G, m, n, steps = 5, coeffs = 'quintic', eps = 1e-7) {
  const { a, b, c } = NS_COEFFS[coeffs] || NS_COEFFS.quintic;
  let X = Float64Array.from(G);
  let norm = 0;
  for (let i = 0; i < X.length; i++) norm += X[i] * X[i];
  norm = Math.sqrt(norm) + eps;
  for (let i = 0; i < X.length; i++) X[i] /= norm;

  // 若行数 > 列数，转置后迭代（A = X Xᵀ 取较小一侧，省算力且数值更稳）
  const transposed = m > n;
  if (transposed) X = denseT(X, m, n);
  const M = transposed ? n : m;
  const N = transposed ? m : n;

  for (let s = 0; s < steps; s++) {
    const Xt = denseT(X, M, N);                  // Xᵀ [N,M]
    const A = denseMul(X, Xt, M, N, M);          // A = X Xᵀ  [M,M]
    const A2 = denseMul(A, A, M, M, M);          // A²
    const BX = new Float64Array(M * N);
    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let p = 0; p < M; p++) acc += (b * A[i * M + p] + c * A2[i * M + p]) * X[p * N + j];
      BX[i * N + j] = acc;
    }
    for (let i = 0; i < X.length; i++) X[i] = a * X[i] + BX[i];
  }

  if (transposed) X = denseT(X, M, N);
  return X;
}

/** Moonlight 缩放：0.2·√max(dout,din)，让 Muon 步长 RMS 与 Adam 对齐 */
export function moonlightScale(dout, din) {
  return 0.2 * Math.sqrt(Math.max(dout, din));
}

export class Muon {
  /**
   * @param {object[]} params [{tensor:Tensor, kind:'muon'|'adamw'}]
   */
  constructor(params, opts = {}) {
    this.lr = opts.lr ?? 0.02;
    this.momentum = opts.momentum ?? 0.95;
    this.nsSteps = opts.nsSteps ?? 5;
    this.nsCoeffs = opts.nsCoeffs ?? 'quintic';
    this.weightDecay = opts.weightDecay ?? 0.01;
    this.nesterov = opts.nesterov ?? false;
    this.t = 0;
    this.groups = { muon: [], adamw: [] };
    this.state = new Map();
    for (const p of params) {
      const kind = p.kind || 'adamw';
      this.groups[kind].push(p.tensor);
      this.state.set(p.tensor, { buf: null, m: null, v: null });
    }
  }

  zeroGrad() {
    for (const g of [this.groups.muon, this.groups.adamw]) for (const p of g) p.zeroGrad();
  }

  step() {
    this.t++;
    const { lr, momentum, wd } = { lr: this.lr, momentum: this.momentum, wd: this.weightDecay };

    /* ---- ① Muon：2D 矩阵 ---- */
    for (const p of this.groups.muon) {
      if (!p.grad) continue;
      const [dout, din] = p.shape;
      const st = this.state.get(p);
      if (!st.buf) st.buf = new Float64Array(p.size);
      const g = p.grad, buf = st.buf;
      for (let i = 0; i < p.size; i++) buf[i] = momentum * buf[i] + g[i];
      const M = this.nesterov
        ? Float64Array.from({ length: p.size }, (_, i) => g[i] + momentum * buf[i])
        : buf;
      const O = newtonSchulz(M, dout, din, this.nsSteps, this.nsCoeffs);
      // 解耦权重衰减
      if (wd > 0) for (let i = 0; i < p.size; i++) p.data[i] *= (1 - lr * wd);
      const s = lr * moonlightScale(dout, din);
      for (let i = 0; i < p.size; i++) p.data[i] -= s * O[i];
    }

    /* ---- ② AdamW：1D / 嵌入 / 输出头 ---- */
    const b1 = 0.9, b2 = 0.95, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (const p of this.groups.adamw) {
      if (!p.grad) continue;
      const st = this.state.get(p);
      if (!st.m) { st.m = new Float64Array(p.size); st.v = new Float64Array(p.size); }
      if (wd > 0) for (let i = 0; i < p.size; i++) p.data[i] *= (1 - lr * wd);
      for (let i = 0; i < p.size; i++) {
        const g = p.grad[i];
        st.m[i] = b1 * st.m[i] + (1 - b1) * g;
        st.v[i] = b2 * st.v[i] + (1 - b2) * g * g;
        const mh = st.m[i] / c1, vh = st.v[i] / c2;
        p.data[i] -= lr * mh / (Math.sqrt(vh) + eps);
      }
    }
  }
}

/* ------------------------------------------------------------------ *
 *  学习率调度：余弦退火 + 线性 warmup（Chinchilla / GPT-3 配方）
 * ------------------------------------------------------------------ */
export function cosineSchedule(step, totalSteps, warmup, baseLr, minFrac = 0.1) {
  if (step < warmup) return baseLr * (step + 1) / warmup;
  const prog = (step - warmup) / Math.max(1, totalSteps - warmup);
  const cos = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, prog)));
  return baseLr * (minFrac + (1 - minFrac) * cos);
}
