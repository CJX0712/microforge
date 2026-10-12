/**
 * forge.js — microforge 统一入口：模型 + 优化器 + 分词器 + 训练循环
 *
 * 设计原则（对应本项目三大信条）：
 *   零重写 —— 每个组件都直接对应一篇顶级论文 / 一个顶级开源实现（见 gpt.js 头部血统表）
 *   集大成 —— 组件之间按工业界最佳实践组合：Pre-LN + RoPE + GQA + SwiGLU + FlashAttn-2 + Muon
 *   可验证 —— 全部由 tests/invariants.js 的 10 条硬不变量交叉钉死
 *
 * @author 晨星
 */

import { clearTape, mulberry32, Tensor } from './tensor.js';
import { GPT, generate, sample, DEFAULT_CONFIG } from './gpt.js';
import { Muon, newtonSchulz, moonlightScale, cosineSchedule } from './muon.js';
import { BPETokenizer } from './bpe.js';
import { CORPUS, corpusStats } from './corpus.js';
import { attentionNaive, attentionFlash, gqaMap } from './flash.js';

export * as T from './tensor.js';
export {
  GPT, generate, sample, DEFAULT_CONFIG,
  Muon, newtonSchulz, moonlightScale, cosineSchedule,
  BPETokenizer, CORPUS, corpusStats,
  attentionNaive, attentionFlash, gqaMap,
  clearTape, mulberry32, Tensor,
};

export const VERSION = '1.0.0';
export const AUTHOR = '晨星';

/* ------------------------------------------------------------------ *
 *  数据管道
 * ------------------------------------------------------------------ */

/** 语料 → 分词 → 展平 token 流 */
export function tokenizeCorpus(text, nMerges = 400, verbose = false) {
  const tok = new BPETokenizer();
  tok.train(text, nMerges, verbose);
  const ids = tok.encode(text);
  return { tok, ids };
}

/** 构造 (x, y) 训练批：y 是 x 右移一位（标准自回归语言建模） */
export function makeBatch(ids, seqLen, batchSize, rng) {
  const xs = new Int32Array(batchSize * seqLen);
  const ys = new Int32Array(batchSize * seqLen);
  const maxStart = ids.length - seqLen - 1;
  for (let b = 0; b < batchSize; b++) {
    const start = Math.floor(rng() * Math.max(1, maxStart));
    for (let t = 0; t < seqLen; t++) {
      xs[b * seqLen + t] = ids[start + t];
      ys[b * seqLen + t] = ids[start + t + 1];
    }
  }
  // 返回 [batchSize][seqLen] 的二维数组，逐样本前向（本项目 seq/batch 都很小）
  const X = [], Y = [];
  for (let b = 0; b < batchSize; b++) {
    X.push(Array.from(xs.subarray(b * seqLen, (b + 1) * seqLen)));
    Y.push(Array.from(ys.subarray(b * seqLen, (b + 1) * seqLen)));
  }
  return { X, Y };
}

/* ------------------------------------------------------------------ *
 *  训练
 * ------------------------------------------------------------------ */

/**
 * 单步训练（前向 → 反向 → 优化器更新）
 * 每个样本独立前向，梯度在参数上累加 ⇒ 等价于 batch 平均
 */
export function trainStep(model, opt, X, Y) {
  clearTape();
  opt.zeroGrad();
  let total = 0;
  for (let b = 0; b < X.length; b++) {
    const loss = model.forward(X[b], Y[b]);
    loss.backward();
    total += loss.data[0];
  }
  // 梯度按 batch 平均
  const inv = 1 / X.length;
  for (const p of opt.groups.muon) if (p.grad) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= inv;
  for (const p of opt.groups.adamw) if (p.grad) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= inv;
  opt.step();
  return total / X.length;
}

/** 评估损失（不更新参数） */
export function evalLoss(model, X, Y) {
  let total = 0;
  const saved = [];
  for (const { tensor: p } of model.params()) { saved.push(p.requiresGrad); p.requiresGrad = false; }
  for (let b = 0; b < X.length; b++) {
    clearTape();
    const loss = model.forward(X[b], Y[b]);
    total += loss.data[0];
  }
  let i = 0;
  for (const { tensor: p } of model.params()) { p.requiresGrad = saved[i++]; }
  return total / X.length;
}

/**
 * 完整训练流程
 * @param {object} opts { text, nMerges, seqLen, batchSize, steps, lr, seed, onStep, evalEvery }
 */
export function train(opts = {}) {
  const text = opts.text || CORPUS;
  const nMerges = opts.nMerges ?? 420;
  const seqLen = opts.seqLen ?? 64;
  const batchSize = opts.batchSize ?? 8;
  const steps = opts.steps ?? 600;
  const baseLr = opts.lr ?? 0.02;
  const warmup = opts.warmup ?? Math.floor(steps * 0.05);
  const seed = opts.seed ?? 1337;
  const evalEvery = opts.evalEvery ?? Math.max(1, Math.floor(steps / 20));
  const cfg = opts.cfg || {};

  const { tok, ids } = tokenizeCorpus(text, nMerges);
  // 90/10 切分训练/验证
  const cut = Math.floor(ids.length * 0.9);
  const trainIds = ids.slice(0, cut);
  const valIds = ids.slice(cut);

  const model = new GPT(Object.assign({ vocab: tok.size, maxSeq: seqLen }, cfg), seed);
  const opt = new Muon(model.params(), {
    lr: baseLr, momentum: 0.95, nsSteps: 5, weightDecay: 0.01,
  });
  const rng = mulberry32(seed + 1);

  const history = [];
  for (let s = 0; s < steps; s++) {
    opt.lr = cosineSchedule(s, steps, warmup, baseLr);
    const { X, Y } = makeBatch(trainIds, seqLen, batchSize, rng);
    const loss = trainStep(model, opt, X, Y);
    history.push(loss);
    if (opts.onStep) opts.onStep({ step: s, loss, lr: opt.lr });
    if (evalEvery > 0 && (s % evalEvery === 0 || s === steps - 1)) {
      const vx = [], vy = [];
      const n = Math.min(4, Math.max(1, Math.floor((valIds.length - seqLen - 1) / seqLen)));
      for (let b = 0; b < n; b++) {
        const st = b * seqLen;
        vx.push(valIds.slice(st, st + seqLen));
        vy.push(valIds.slice(st + 1, st + seqLen + 1));
      }
      if (vx.length) {
        const vl = evalLoss(model, vx, vy);
        if (opts.onEval) opts.onEval({ step: s, trainLoss: loss, valLoss: vl });
        if (opts.onStep) history[history.length - 1] = loss;
        model._lastVal = vl;
      }
    }
  }
  return { model, tok, trainIds, valIds, history };
}
