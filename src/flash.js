/**
 * flash.js — 注意力双实现：朴素（带反向，用于训练） + FlashAttention-2（online softmax，用于推理/交叉验证）
 *
 * 血统（复用世界顶级成果，非自造）：
 *   - Scaled Dot-Product Attention：Vaswani et al. 2017 (NeurIPS)
 *   - FlashAttention-2 online softmax / running max-ℓ rescaling：Dao 2023 (ICLR 2024)
 *       核心恒等式：O ← O·exp(m_old − m_new) + exp(S_block − m_new)·V_block
 *                   ℓ ← ℓ·exp(m_old − m_new) + rowsum(exp(S_block − m_new))
 *       这条恒等式就是「分块 == 全局」的数学证明，tests/invariants.js 用数值把它钉死。
 *   - Causal masking：GPT 自回归标配
 *
 * 两条路径互为交叉验证：max|Flash − Naive| 必须 < 1e-5（Float32 舍入量级）。
 *
 * @author 晨星
 */

import { Tensor, TAPE } from './tensor.js';

/**
 * 朴素因果注意力（保存 softmax 矩阵 → 可反向）
 * @param {Tensor} q [B,H,T,dh]
 * @param {Tensor} k [B,H,T,dh]
 * @param {Tensor} v [B,H,T,dv]
 * @param {number} smScale 1/sqrt(dh)
 * @param {boolean} causal
 * @returns {Tensor} [B,H,T,dv]
 */
export function attentionNaive(q, k, v, smScale, causal = true) {
  const [B, H, T, dh] = q.shape;
  const dv = v.shape[3];
  const out = new Float32Array(B * H * T * dv);
  const P = new Float32Array(B * H * T * T);   // softmax 权重，反向要用
  const Q = q.data, K = k.data, V = v.data;

  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
    const base = (b * H + h);
    for (let i = 0; i < T; i++) {
      const pi = base * T * T + i * T;
      let mx = -Infinity;
      for (let j = 0; j < T; j++) {
        if (causal && j > i) continue;
        let s = 0;
        for (let d = 0; d < dh; d++) s += Q[(base * T + i) * dh + d] * K[(base * T + j) * dh + d];
        s *= smScale;
        P[pi + j] = s;
        if (s > mx) mx = s;
      }
      let sum = 0;
      const lim = causal ? i + 1 : T;
      for (let j = 0; j < lim; j++) { const e = Math.exp(P[pi + j] - mx); P[pi + j] = e; sum += e; }
      const inv = 1 / sum;
      for (let j = 0; j < lim; j++) P[pi + j] *= inv;
      for (let d = 0; d < dv; d++) {
        let acc = 0;
        for (let j = 0; j < lim; j++) acc += P[pi + j] * V[(base * T + j) * dv + d];
        out[(base * T + i) * dv + d] = acc;
      }
    }
  }

  const o = new Tensor(out, [B, H, T, dv]);
  if (q.requiresGrad || k.requiresGrad || v.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      const G = o.grad;
      for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
        const base = b * H + h;
        for (let i = 0; i < T; i++) {
          const pi = base * T * T + i * T;
          const lim = causal ? i + 1 : T;
          // ① dP_raw[i,j] = G[i,:] · V[j,:]
          const dP = new Float64Array(lim);
          for (let j = 0; j < lim; j++) {
            let dp = 0;
            for (let d = 0; d < dv; d++) dp += G[(base * T + i) * dv + d] * V[(base * T + j) * dv + d];
            dP[j] = dp;
          }
          // ② softmax 反向：dot = Σ_j P[i,j]·dP_raw[i,j] ; dS = P·(dP_raw − dot)
          let dot = 0;
          for (let j = 0; j < lim; j++) dot += P[pi + j] * dP[j];
          // ③ 累加到 V / Q / K
          for (let j = 0; j < lim; j++) {
            const p = P[pi + j];
            const ds = p * (dP[j] - dot) * smScale;
            for (let d = 0; d < dv; d++) v.grad[(base * T + j) * dv + d] += p * G[(base * T + i) * dv + d];
            for (let d = 0; d < dh; d++) {
              q.grad[(base * T + i) * dh + d] += ds * K[(base * T + j) * dh + d];
              k.grad[(base * T + j) * dh + d] += ds * Q[(base * T + i) * dh + d];
            }
          }
        }
      }
    };
  }
  return o;
}

/**
 * FlashAttention-2 风格前向：online softmax + 分块，不物化 T×T 矩阵
 * 复杂度仍是 O(T²) FLOPs，但 HBM 访问为 O(T) —— 这里主要作为
 *   (a) 数值等价性不变量的对照实现
 *   (b) KV-cache 增量解码的实际执行路径
 *
 * @param {Tensor} q [B,H,Tq,dh]
 * @param {Tensor} k [B,H,Tk,dh]
 * @param {Tensor} v [B,H,Tk,dv]
 * @param {number} smScale
 * @param {boolean} causal  若为 true，按 q 的绝对位置 i 与 k 的绝对位置 j 判断 j<=i
 * @param {number} blockSize KV 分块大小
 * @param {number} qOffset   q 在整条序列中的起始位置（增量解码用）
 * @returns {Float32Array} [B,H,Tq,dv]
 */
export function attentionFlash(q, k, v, smScale, causal = true, blockSize = 32, qOffset = 0) {
  const [B, H, Tq, dh] = q.shape;
  const Tk = k.shape[2], dv = v.shape[3];
  const out = new Float32Array(B * H * Tq * dv);
  const Q = q.data, K = k.data, V = v.data;

  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
    const base = b * H + h;
    for (let i = 0; i < Tq; i++) {
      const gi = qOffset + i;                       // 该 query 的绝对位置
      let m = -Infinity, l = 0;
      const acc = new Float64Array(dv);
      const kbEnd = causal ? Math.min(Tk, gi + 1) : Tk;
      for (let j0 = 0; j0 < kbEnd; j0 += blockSize) {
        const j1 = Math.min(j0 + blockSize, kbEnd);
        // 块内先算原始分数，取块最大值
        const blk = (j1 - j0);
        const sBlk = new Float64Array(blk);
        let mBlk = -Infinity;
        for (let jj = 0; jj < blk; jj++) {
          const j = j0 + jj;
          let s = 0;
          for (let d = 0; d < dh; d++) s += Q[(base * Tq + i) * dh + d] * K[(base * Tk + j) * dh + d];
          s *= smScale;
          sBlk[jj] = s;
          if (s > mBlk) mBlk = s;
        }
        const mNew = Math.max(m, mBlk);
        const corrOld = (m === -Infinity) ? 0 : Math.exp(m - mNew);
        const p = new Float64Array(blk);
        let lBlk = 0;
        for (let jj = 0; jj < blk; jj++) {
          const e = Math.exp(sBlk[jj] - mNew);
          p[jj] = e; lBlk += e;
        }
        for (let d = 0; d < dv; d++) acc[d] *= corrOld;
        for (let jj = 0; jj < blk; jj++) {
          const j = j0 + jj, e = p[jj];
          for (let d = 0; d < dv; d++) acc[d] += e * V[(base * Tk + j) * dv + d];
        }
        l = l * corrOld + lBlk;
        m = mNew;
      }
      const inv = l === 0 ? 0 : 1 / l;
      for (let d = 0; d < dv; d++) out[(base * Tq + i) * dv + d] = acc[d] * inv;
    }
  }
  return out;
}

/**
 * GQA 分组映射：第 i 个 query 头使用第 (i * nKV / nQ) 个 kv 头
 * 血统：Ainslie et al. 2023 (GQA: Training Generalized Multi-Query Transformer Models)
 * 不变量：nKV === nQ 时退化为 MHA，数值必须与 MHA 完全一致
 */
export function gqaMap(i, nQ, nKV) {
  return Math.floor((i * nKV) / nQ);
}
