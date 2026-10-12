/**
 * gpt.js — 微型 GPT 模型：现代 LLM 全部关键组件的手写实现
 *
 * 组件血统（每一项都是当前工业界事实标准，复用而非自造）：
 *   ┌ 组件           ┌ 论文 / 来源                                        ┌ 年份
 *   │ Pre-LN RMSNorm │ Zhang & Sennrich, "Root Mean Square Layer Norm"    │ 2019
 *   │ RoPE           │ Su et al., "RoFormer: Enhanced Transformer with
 *   │                │   Rotary Position Embedding"                       │ 2021
 *   │ GQA            │ Ainslie et al., "GQA: Training Generalized
 *   │                │   Multi-Query Transformer Models" (EMNLP)          │ 2023
 *   │ SwiGLU         │ Shazeer, "GLU Variants Improve Transformer"        │ 2020
 *   │ FlashAttn-2    │ Dao, "FlashAttention-2" (ICLR)                     │ 2024
 *   │ Tied Embedding │ Press & Wolf, "Using the Output Embedding to
 *   │                │   Improve Language Models" (EACL)                  │ 2017
 *   │ Scaled Init    │ Radford et al., GPT-2 (残差投影 1/√(2·L) 缩放)      │ 2019
 *   └ KV-Cache       └ 自回归解码标配                                     └ —
 *
 * @author 晨星
 */

import {
  Tensor, TAPE, mulberry32,
  matmul, linear, add, mul, silu, rmsnorm, rope, embedding,
  crossEntropy, transpose, reshape, splitHeads, mergeHeads, sliceLast,
} from './tensor.js';
import { attentionNaive, attentionFlash, gqaMap } from './flash.js';

export const DEFAULT_CONFIG = {
  vocab: 512,
  dModel: 96,
  nHeads: 6,
  nKVHeads: 2,      // GQA：6 query 头共享 2 组 K/V ⇒ KV-cache 缩小 3×
  nLayers: 4,
  dFF: 256,
  maxSeq: 128,
  ropeBase: 10000,
};

export class GPT {
  constructor(cfg = {}, seed = 1337) {
    this.cfg = Object.assign({}, DEFAULT_CONFIG, cfg);
    const C = this.cfg;
    if (C.dModel % C.nHeads !== 0) throw new Error('dModel 必须能被 nHeads 整除');
    if (C.nHeads % C.nKVHeads !== 0) throw new Error('nHeads 必须能被 nKVHeads 整除');
    this.dh = C.dModel / C.nHeads;
    this.rng = mulberry32(seed);
    this._init();
  }

  _init() {
    const C = this.cfg, D = C.dModel, dh = this.dh;
    const H = C.nHeads, HKV = C.nKVHeads, F = C.dFF, L = C.nLayers;
    const R = this.rng;
    const resScale = 1 / Math.sqrt(2 * L);          // GPT-2 残差缩放

    const mk = (dout, din, std) => Tensor.randn([dout, din], R, std).requireGrad();
    const mk1 = (n, v = 1) => { const t = Tensor.zeros([n]); t.data.fill(v); return t.requireGrad(); };

    this.p = { layers: [] };
    this.p.tokEmb = Tensor.randn([C.vocab, D], R, 0.02).requireGrad();

    for (let l = 0; l < L; l++) {
      const layer = {
        wIn: mk1(D),                                   // RMSNorm 权重（1D → AdamW）
        wq: mk(H * dh, D, Math.sqrt(1 / D)),           // 2D → Muon
        wk: mk(HKV * dh, D, Math.sqrt(1 / D)),
        wv: mk(HKV * dh, D, Math.sqrt(1 / D)),
        wo: mk(D, H * dh, Math.sqrt(1 / (H * dh)) * resScale),
        wPost: mk1(D),
        wGate: mk(F, D, Math.sqrt(1 / D)),
        wUp: mk(F, D, Math.sqrt(1 / D)),
        wDown: mk(D, F, Math.sqrt(1 / F) * resScale),
      };
      layer.wq.name = `L${l}.wq`; layer.wk.name = `L${l}.wk`; layer.wv.name = `L${l}.wv`;
      layer.wo.name = `L${l}.wo`; layer.wGate.name = `L${l}.wGate`;
      layer.wUp.name = `L${l}.wUp`; layer.wDown.name = `L${l}.wDown`;
      this.p.layers.push(layer);
    }
    this.p.finalNorm = mk1(D);
  }

  /** 参数表：标记 Muon / AdamW 归属（Moonlight 论文 3.5 节工程共识） */
  params() {
    const out = [{ tensor: this.p.tokEmb, kind: 'adamw' }];
    for (const l of this.p.layers) {
      out.push({ tensor: l.wIn, kind: 'adamw' });
      out.push({ tensor: l.wPost, kind: 'adamw' });
      for (const k of ['wq', 'wk', 'wv', 'wo', 'wGate', 'wUp', 'wDown']) out.push({ tensor: l[k], kind: 'muon' });
    }
    out.push({ tensor: this.p.finalNorm, kind: 'adamw' });
    return out;
  }

  paramCount() {
    let n = 0; for (const { tensor: t } of this.params()) n += t.size;
    return n;
  }

  /**
   * 前向 +（可选）交叉熵损失
   * @param {Int32Array|number[]} ids 输入 token ids，长度 T
   * @param {Int32Array|number[]} [targets] 目标 ids；给定则返回 loss
   * @param {{posOffset?:number, collect?:object}} [opts]
   */
  forward(ids, targets = null, opts = {}) {
    const C = this.cfg, D = C.dModel, H = C.nHeads, HKV = C.nKVHeads, dh = this.dh;
    const T = ids.length;
    const smScale = 1 / Math.sqrt(dh);
    const posOff = opts.posOffset || 0;

    let x = embedding(this.p.tokEmb, ids);            // [T,D]
    x = reshape(x, [1, T, D]);

    for (const l of this.p.layers) {
      /* --- Pre-LN 分支 1：GQA 多头注意力 --- */
      const h = rmsnorm(x, l.wIn);                    // [1,T,D]
      const q = linear(reshape(h, [T, D]), l.wq);     // [T, H*dh]
      const k = linear(reshape(h, [T, D]), l.wk);     // [T, HKV*dh]
      const v = linear(reshape(h, [T, D]), l.wv);

      const q4 = splitHeads(reshape(q, [1, T, H * dh]), H);   // [1,H,T,dh]
      const k4 = splitHeads(reshape(k, [1, T, HKV * dh]), HKV);// [1,HKV,T,dh]
      const v4 = splitHeads(reshape(v, [1, T, HKV * dh]), HKV);

      const qr = rope(q4, C.ropeBase, posOff);
      const kr = rope(k4, C.ropeBase, posOff);

      // GQA 展开：把 HKV 组 K/V 广播到 H 个 query 头
      const kExp = expandGQA(kr, H, HKV);
      const vExp = expandGQA(v4, H, HKV);

      const att = attentionNaive(qr, kExp, vExp, smScale, true);  // [1,H,T,dh]
      const attMerged = mergeHeads(att);                          // [1,T,D]
      const proj = linear(reshape(attMerged, [T, D]), l.wo);      // [T,D]
      const x2 = add(x, reshape(proj, [1, T, D]));

      /* --- Pre-LN 分支 2：SwiGLU FFN --- */
      const h2 = rmsnorm(x2, l.wPost);
      const g = silu(linear(reshape(h2, [T, D]), l.wGate));       // [T,F]
      const u = linear(reshape(h2, [T, D]), l.wUp);
      const fu = mul(g, u);
      const down = linear(fu, l.wDown);                           // [T,D]
      x = add(x2, reshape(down, [1, T, D]));
    }

    const xf = rmsnorm(x, this.p.finalNorm);
    const logits = linear(reshape(xf, [T, D]), this.p.tokEmb);   // 权重绑定

    if (targets == null) return logits;
    return crossEntropy(logits, targets);
  }
}

/** GQA 展开：[1,HKV,T,dh] → [1,H,T,dh]，第 i 个 q 头用第 gqaMap(i) 组 KV */
function expandGQA(kv, H, HKV) {
  if (H === HKV) return kv;
  const [B, , T, dh] = kv.shape;
  const out = new Float32Array(B * H * T * dh);
  for (let b = 0; b < B; b++) for (let i = 0; i < H; i++) {
    const src = b * HKV + gqaMap(i, H, HKV);
    out.set(kv.data.subarray(src * T * dh, (src + 1) * T * dh), (b * H + i) * T * dh);
  }
  const o = new Tensor(out, [B, H, T, dh]);
  if (kv.requiresGrad) {
    o.requireGrad(); TAPE.push(o);
    o._back = () => {
      for (let b = 0; b < B; b++) for (let i = 0; i < H; i++) {
        const src = b * HKV + gqaMap(i, H, HKV);
        for (let j = 0; j < T * dh; j++) kv.grad[src * T * dh + j] += o.grad[(b * H + i) * T * dh + j];
      }
    };
  }
  return o;
}

/* ------------------------------------------------------------------ *
 *  KV-Cache 增量解码：FlashAttention-2 路径，单层单 token
 * ------------------------------------------------------------------ */

/** [1, Din] × [Dout, Din] → [1, Dout]（无图，纯前向） */
function matvec(W, x, dout, din) {
  const out = new Float64Array(dout);
  const w = W.data;
  for (let i = 0; i < dout; i++) {
    let s = 0;
    const oi = i * din;
    for (let j = 0; j < din; j++) s += w[oi + j] * x[j];
    out[i] = s;
  }
  return out;
}

function rmsnormVec(x, w, eps = 1e-6) {
  const D = x.length;
  let ss = 0; for (let i = 0; i < D; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / D + eps);
  const out = new Float64Array(D);
  for (let i = 0; i < D; i++) out[i] = x[i] * inv * w.data[i];
  return out;
}

function splitHeadsRaw(x, H, dh) {
  // x: Float64Array[D] → Float32Array[H*dh]，布局 [h][dh]
  const out = new Float32Array(H * dh);
  for (let h = 0; h < H; h++) for (let d = 0; d < dh; d++) out[h * dh + d] = x[h * dh + d];
  return out;
}

function ropeVec(x, H, dh, pos, base) {
  const out = new Float32Array(x.length);
  for (let h = 0; h < H; h++) for (let k = 0; k < dh / 2; k++) {
    const theta = pos / Math.pow(base, (2 * k) / dh);
    const c = Math.cos(theta), s = Math.sin(theta);
    const i0 = h * dh + 2 * k, i1 = i0 + 1;
    const a = x[i0], b = x[i1];
    out[i0] = a * c - b * s;
    out[i1] = a * s + b * c;
  }
  return out;
}

/**
 * 自回归生成（KV-cache + FlashAttention-2 + top-k / temperature 采样）
 * @param {GPT} model
 * @param {number[]} promptIds
 * @param {number} maxNew
 * @param {object} opts { temperature, topK, rng, stopIds }
 */
export function generate(model, promptIds, maxNew = 120, opts = {}) {
  const C = model.cfg, D = C.dModel, H = C.nHeads, HKV = C.nKVHeads, dh = model.dh;
  const F = C.dFF, L = C.nLayers;
  const temp = opts.temperature ?? 0.9;
  const topK = opts.topK ?? 40;
  const rng = opts.rng || Math.random;
  const smScale = 1 / Math.sqrt(dh);

  // KV-cache: 每层 [HKV][maxLen][dh]，stride = maxLen*dh
  const maxLen = promptIds.length + maxNew;
  const stride = maxLen * dh;
  const cache = [];
  for (let l = 0; l < L; l++) cache.push({
    k: new Float32Array(HKV * stride),
    v: new Float32Array(HKV * stride),
    len: 0,
  });

  const seq = promptIds.slice();
  const out = [];
  let firstLogits = null;

  /**
   * 单步：把 seq[pos] 送进网络，写入 KV-cache 第 pos 槽，
   * 用 FlashAttention-2 对 cache[0..pos] 做注意力，返回该位置的 logits。
   * 因果性天然成立：此刻 cache 里只有 0..pos，不会看到未来。
   */
  function stepToken(pos) {
    const id = seq[pos];
    const xv = new Float64Array(D);
    const emb = model.p.tokEmb.data;
    for (let j = 0; j < D; j++) xv[j] = emb[id * D + j];

    for (let l = 0; l < L; l++) {
      const lay = model.p.layers[l];
      const h = rmsnormVec(xv, lay.wIn);
      const qv = matvec(lay.wq, h, H * dh, D);
      const kv = matvec(lay.wk, h, HKV * dh, D);
      const vv = matvec(lay.wv, h, HKV * dh, D);
      const qh = ropeVec(splitHeadsRaw(qv, H, dh), H, dh, pos, C.ropeBase);
      const kh = ropeVec(splitHeadsRaw(kv, HKV, dh), HKV, dh, pos, C.ropeBase);
      const vh = splitHeadsRaw(vv, HKV, dh);

      // 写入 cache 第 pos 槽
      const ck = cache[l];
      for (let g = 0; g < HKV; g++) {
        ck.k.set(kh.subarray(g * dh, (g + 1) * dh), g * stride + pos * dh);
        ck.v.set(vh.subarray(g * dh, (g + 1) * dh), g * stride + pos * dh);
      }
      ck.len = pos + 1;

      // FlashAttention-2（分块 online softmax）：q 长度 1，kv 长度 pos+1
      const attn = new Float64Array(D);
      for (let i = 0; i < H; i++) {
        const g = gqaMap(i, H, HKV);
        const q1 = new Tensor(qh.subarray(i * dh, (i + 1) * dh), [1, 1, 1, dh]);
        const k1 = new Tensor(ck.k.subarray(g * stride, (g + 1) * stride), [1, 1, ck.len, dh]);
        const v1 = new Tensor(ck.v.subarray(g * stride, (g + 1) * stride), [1, 1, ck.len, dh]);
        const o = attentionFlash(q1, k1, v1, smScale, false, 32, 0);
        for (let d = 0; d < dh; d++) attn[i * dh + d] = o[d];
      }

      const proj = matvec(lay.wo, attn, D, H * dh);
      for (let j = 0; j < D; j++) xv[j] += proj[j];

      const h2 = rmsnormVec(xv, lay.wPost);
      const gv = matvec(lay.wGate, h2, F, D);
      const uv = matvec(lay.wUp, h2, F, D);
      for (let j = 0; j < F; j++) gv[j] = (gv[j] / (1 + Math.exp(-gv[j]))) * uv[j];   // SwiGLU
      const dv = matvec(lay.wDown, gv, D, F);
      for (let j = 0; j < D; j++) xv[j] += dv[j];
    }

    const xf = rmsnormVec(xv, model.p.finalNorm);
    const logits = new Float64Array(C.vocab);
    const E = model.p.tokEmb.data;
    for (let t = 0; t < C.vocab; t++) {
      let s = 0; const oi = t * D;
      for (let j = 0; j < D; j++) s += E[oi + j] * xf[j];
      logits[t] = s;
    }
    return logits;
  }

  // 阶段一 prefill：逐位置填满 KV-cache（等价于一次性前向整个 prompt）
  let logits = null;
  for (let p = 0; p < promptIds.length; p++) logits = stepToken(p);

  // 阶段二 decode：每步采样一个 token，再把它送回网络
  for (let step = 0; step < maxNew; step++) {
    if (step === 0 && opts.collect) firstLogits = logits;
    const nextId = sample(logits, temp, topK, rng);
    seq.push(nextId);
    out.push(nextId);
    if (opts.stopIds && opts.stopIds.includes(nextId)) break;
    logits = stepToken(seq.length - 1);
  }
  if (opts.collect) opts.collect.firstLogits = firstLogits;
  return out;
}

/** temperature + top-k 采样（GPT-2 / Holtzman et al. 2020 top-p 的近亲） */
export function sample(logits, temperature, topK, rng) {
  if (temperature <= 0) {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; }
    return bi;
  }
  const scaled = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  const idx = Array.from(scaled.keys()).sort((a, b) => scaled[b] - scaled[a]).slice(0, topK);
  let mx = -Infinity;
  for (const i of idx) if (scaled[i] > mx) mx = scaled[i];
  let sum = 0;
  const probs = new Float64Array(idx.length);
  for (let j = 0; j < idx.length; j++) { probs[j] = Math.exp(scaled[idx[j]] - mx); sum += probs[j]; }
  let r = rng() * sum;
  for (let j = 0; j < idx.length; j++) { r -= probs[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}

export { expandGQA, matvec, rmsnormVec, sliceLast };
