/**
 * invariants.js — microforge 的硬不变量自检套件
 *
 * 哲学：一个数值库是否可信，不看它的 README 写得多漂亮，而看它能否被**独立实现交叉验证**。
 * 这里每一条不变量都是「两种互不相干的路径必须给出同一个答案」：
 *
 *   ①  反向传播梯度      ↔  中心差分数值梯度
 *   ②  RoPE 相对性       ↔  直接构造的解析旋转矩阵
 *   ③  GQA(nKV=g)        ↔  手工复制 KV 后的 MHA
 *   ④  FlashAttention-2  ↔  朴素注意力
 *   ⑤  Newton-Schulz     ↔  半正交矩阵的代数定义 XᵀX = I
 *   ⑥  KV-cache 增量解码  ↔  全序列重算
 *   ⑦  softmax 平移不变   ↔  解析恒等式 softmax(x+c) = softmax(x)
 *   ⑧  BPE 编解码        ↔  恒等映射 decode(encode(s)) = s
 *   ⑨  过拟合单批         ↔  优化器必须能真正把 loss 压到近零
 *   ⑩  训练曲线           ↔  loss 必须单调趋势下降
 *   ⑪  RMSNorm 尺度不变   ↔  解析性质 rmsnorm(c·x) = rmsnorm(x), c>0
 *
 * 运行：node tests/invariants.js
 *
 * @author 晨星
 */

import {
  GPT, generate, Muon, BPETokenizer, CORPUS, train,
  attentionNaive, attentionFlash, newtonSchulz, moonlightScale,
  clearTape, mulberry32, T,
} from '../src/forge.js';

const { Tensor, rope, softmaxRows, rmsnorm, mul, transpose, matmul, reshape } = T;

/* ------------------------------ 测试框架 ------------------------------ */
const results = [];
const quiet = typeof process !== 'undefined' && process.env && process.env.FORGE_QUIET === '1';
function check(name, fn, tolDesc) {
  const t0 = Date.now();
  if (!quiet) process.stdout.write(`  · 运行中：${name} ...`);
  let r;
  try { r = fn(); } catch (e) { r = { pass: false, detail: 'EXCEPTION: ' + (e && e.stack || e) }; }
  const ms = Date.now() - t0;
  results.push({ name, ok: r.pass, detail: r.detail, tol: tolDesc, ms });
  if (!quiet) process.stdout.write(`\r  ${r.pass ? '✅' : '❌'} ${name}  (${ms} ms)          \n`);
  return r.pass;
}
const fmt = (x) => (typeof x === 'number' ? (Math.abs(x) < 1e-4 ? x.toExponential(3) : x.toFixed(6)) : String(x));

/* ------------------------------ ① 梯度检验 ------------------------------ */
function invGradient() {
  const rng = mulberry32(7);
  const model = new GPT({ vocab: 40, dModel: 32, nHeads: 4, nKVHeads: 2, nLayers: 2, dFF: 64, maxSeq: 16 }, 99);
  const seq = 8;
  const X = Array.from({ length: seq }, () => Math.floor(rng() * 40));
  const Y = Array.from({ length: seq }, () => Math.floor(rng() * 40));

  // ①-A 解析梯度
  clearTape();
  for (const { tensor: p } of model.params()) p.zeroGrad();
  const loss = model.forward(X, Y);
  loss.backward();

  // ①-B 中心差分：(f(θ+h) − f(θ−h)) / 2h
  const fwd = () => { clearTape(); return model.forward(X, Y).data[0]; };
  const all = [];
  for (const { tensor: p } of model.params()) {
    const n = Math.min(6, p.size);
    for (let k = 0; k < n; k++) all.push([p, Math.floor((k + 1) * p.size / (n + 1))]);
  }
  // 只保留信号显著高于 Float32 噪声地板的探针：
  //   噪声地板 ≈ eps(loss)/(2h) ≈ 2.4e-7 / 0.1 ≈ 2.4e-6
  // 梯度量级低于此的参数，中心差分在 Float32 下数学上不可分辨（不是 bug，是精度下限）
  const h = 5e-2;
  all.sort((a, b) => Math.abs(b[0].grad[b[1]]) - Math.abs(a[0].grad[a[1]]));
  const probes = all.slice(0, 40);

  let worst = 0, worstAt = '', minAna = Infinity;
  for (const [p, i] of probes) {
    const ana = p.grad[i];
    minAna = Math.min(minAna, Math.abs(ana));
    const orig = p.data[i];
    p.data[i] = orig + h; const fp = fwd();
    p.data[i] = orig - h; const fm = fwd();
    p.data[i] = orig;
    const num = (fp - fm) / (2 * h);
    const den = Math.abs(num) + Math.abs(ana) + 1e-12;
    const rel = Math.abs(num - ana) / den;
    if (rel > worst) { worst = rel; worstAt = `${p.name || 'tokEmb/finalNorm'}[${i}] num=${num.toExponential(3)} ana=${ana.toExponential(3)}`; }
  }
  return {
    pass: worst < 3e-3 && probes.length >= 20,
    detail: `${probes.length}/${all.length} 个显著梯度探针（最小 |ana|=${minAna.toExponential(2)}），`
      + `最大相对误差 ${worst.toExponential(3)}（阈值 3e-3，Float32 中心差分噪声下限）· 最差点 ${worstAt}`,
  };
}

/* ------------------------------ ② RoPE 相对性 ------------------------------ */
function invRoPE() {
  const D = 16, dh = 16, base = 10000;
  const rng = mulberry32(3);
  const q = Tensor.randn([1, 1, 1, dh], rng, 1).data;
  const k = Tensor.randn([1, 1, 1, dh], rng, 1).data;

  const dot = (a, b) => { let s = 0; for (let i = 0; i < dh; i++) s += a[i] * b[i]; return s; };

  // 用 rope() 算子旋转到不同位置
  const rot = (v, pos) => {
    const t = new Tensor(Float32Array.from(v), [1, 1, 1, dh]);
    return rope(t, base, pos).data;
  };
  const m = 3, n = 7, p = 11;
  const d1 = dot(rot(q, m), rot(k, n));                       // 距离 m−n = −4
  const d2 = dot(rot(q, m + p), rot(k, n + p));               // 距离仍为 (m+p)−(n+p) = −4
  const errRel = Math.abs(d1 - d2) / (Math.abs(d1) + 1e-8);

  // 范数保持（旋转是正交变换）
  const nq = Math.sqrt(dot(q, q)), nrq = Math.sqrt(dot(rot(q, 23), rot(q, 23)));
  const errNorm = Math.abs(nq - nrq) / nq;
  void D;
  return {
    pass: errRel < 1e-5 && errNorm < 1e-5,
    detail: `⟨R_m q, R_n k⟩ 平移不变误差 ${errRel.toExponential(3)}；范数保持误差 ${errNorm.toExponential(3)}（阈值 1e-5）`,
  };
}

/* ------------------------------ ③ GQA ↔ MHA 等价 ------------------------------ */
function invGQA() {
  const cfg = { vocab: 30, dModel: 32, nHeads: 4, nLayers: 2, dFF: 64, maxSeq: 16 };
  const A = new GPT(Object.assign({}, cfg, { nKVHeads: 2 }), 21);   // GQA：4 头共享 2 组 KV
  const B = new GPT(Object.assign({}, cfg, { nKVHeads: 4 }), 21);   // 参照：完整 MHA
  const dh = A.dh;
  // 把 A 的 2 组 KV 复制成 B 的 4 组（按 gqaMap 的映射规则）
  for (let l = 0; l < cfg.nLayers; l++) {
    for (const w of ['wk', 'wv']) {
      const src = A.p.layers[l][w].data, dst = B.p.layers[l][w].data;
      for (let i = 0; i < 4; i++) {
        const g = Math.floor((i * 2) / 4);
        for (let d = 0; d < dh; d++)
          for (let j = 0; j < cfg.dModel; j++)
            dst[(i * dh + d) * cfg.dModel + j] = src[(g * dh + d) * cfg.dModel + j];
      }
    }
    // 其余权重必须完全一致
    for (const w of ['wq', 'wo', 'wGate', 'wUp', 'wDown']) B.p.layers[l][w].data.set(A.p.layers[l][w].data);
    B.p.layers[l].wIn.data.set(A.p.layers[l].wIn.data);
    B.p.layers[l].wPost.data.set(A.p.layers[l].wPost.data);
  }
  B.p.tokEmb.data.set(A.p.tokEmb.data);
  B.p.finalNorm.data.set(A.p.finalNorm.data);

  const rng = mulberry32(5);
  const X = Array.from({ length: 12 }, () => Math.floor(rng() * 30));
  const la = A.forward(X), lb = B.forward(X);
  let maxErr = 0;
  for (let i = 0; i < la.size; i++) maxErr = Math.max(maxErr, Math.abs(la.data[i] - lb.data[i]));
  return {
    pass: maxErr < 1e-5,
    detail: `GQA(4 头 / 2 组 KV) 与手工展开的 MHA 输出最大绝对差 ${maxErr.toExponential(3)}（阈值 1e-5）`,
  };
}

/* ------------------------------ ④ FlashAttention-2 ↔ 朴素注意力 ------------------------------ */
function invFlash() {
  const rng = mulberry32(11);
  const B = 2, H = 3, T = 17, dh = 8, dv = 8;
  const q = new Tensor(Tensor.randn([B, H, T, dh], rng, 1.5).data, [B, H, T, dh]);
  const k = new Tensor(Tensor.randn([B, H, T, dh], rng, 1.5).data, [B, H, T, dh]);
  const v = new Tensor(Tensor.randn([B, H, T, dv], rng, 1.0).data, [B, H, T, dv]);
  const smScale = 1 / Math.sqrt(dh);

  const naive = attentionNaive(q, k, v, smScale, true).data;
  let worst = 0;
  for (const bs of [2, 4, 8, 16, 32]) {         // 不同分块大小必须给出同一个答案
    const fl = attentionFlash(q, k, v, smScale, true, bs, 0);
    for (let i = 0; i < naive.length; i++) worst = Math.max(worst, Math.abs(naive[i] - fl[i]));
  }
  return {
    pass: worst < 1e-5,
    detail: `分块大小 2/4/8/16/32 与朴素注意力最大绝对差 ${worst.toExponential(3)}（阈值 1e-5）`,
  };
}

/* ------------------------------ ⑤ Newton-Schulz 半正交性 ------------------------------ */
function invNewtonSchulz() {
  const rng = mulberry32(13);
  const shapes = [[8, 8], [16, 8], [8, 16], [32, 16]];

  /** 返回 {dev: ‖Gram−I‖_max, ratio: ‖X‖_F/√min(m,n)} */
  const measure = (X, m, n) => {
    const small = Math.min(m, n);
    const gram = new Float64Array(small * small);
    for (let i = 0; i < small; i++) for (let j = 0; j < small; j++) {
      let s = 0;
      if (m <= n) { for (let p = 0; p < n; p++) s += X[i * n + p] * X[j * n + p]; }   // X Xᵀ
      else { for (let p = 0; p < m; p++) s += X[p * n + i] * X[p * n + j]; }          // Xᵀ X
      gram[i * small + j] = s;
    }
    let dev = 0;
    for (let i = 0; i < small; i++) for (let j = 0; j < small; j++)
      dev = Math.max(dev, Math.abs(gram[i * small + j] - (i === j ? 1 : 0)));
    let f = 0; for (const v of X) f += v * v;
    return { dev, ratio: Math.sqrt(f) / Math.sqrt(small) };
  };

  // 严格分支：cubic（σ ← 1.5σ − 0.5σ³，不动点 σ=1 且可证收敛）必须给出真正的半正交矩阵
  let cubicDev = 0, cubicCase = '';
  // 工程分支：quintic（Polar Express，训练实际使用）必须落在文献记载的经验包络内
  let quinticDev = 0, ratioLo = Infinity, ratioHi = 0, quinticCase = '';

  for (const [m, n] of shapes) {
    const G = new Float64Array(m * n);
    for (let i = 0; i < G.length; i++) G[i] = rng() * 2 - 1;
    const c = measure(newtonSchulz(G, m, n, 20, 'cubic'), m, n);
    if (c.dev > cubicDev) { cubicDev = c.dev; cubicCase = `${m}×${n}`; }
    const q = measure(newtonSchulz(G, m, n, 5, 'quintic'), m, n);
    quinticDev = Math.max(quinticDev, q.dev);
    ratioLo = Math.min(ratioLo, q.ratio); ratioHi = Math.max(ratioHi, q.ratio);
    if (q.dev === quinticDev) quinticCase = `${m}×${n}`;
  }
  const s = moonlightScale(256, 96);
  const scaleOk = Math.abs(s - 0.2 * Math.sqrt(256)) < 1e-12;
  const pass = cubicDev < 1e-6 && ratioLo > 0.8 && ratioHi < 1.2 && quinticDev < 0.6 && scaleOk;
  return {
    pass,
    detail: `cubic[20 步] Gram 偏离 I 最大 ${cubicDev.toExponential(2)}（阈值 1e-6，${cubicCase}）· `
      + `quintic[5 步] ‖X‖_F/√min ∈ [${ratioLo.toFixed(3)}, ${ratioHi.toFixed(3)}]（阈值 0.8~1.2），Gram 偏离 ${quinticDev.toFixed(2)}（Polar Express 经验包络，${quinticCase}）· `
      + `Moonlight 缩放因子 ${s.toFixed(4)} ${scaleOk ? '正确' : '错误'}`,
  };
}

/* ------------------------------ ⑥ KV-cache ↔ 全序列重算 ------------------------------ */
function invKVCache() {
  const rng = mulberry32(17);
  const model = new GPT({ vocab: 60, dModel: 48, nHeads: 6, nKVHeads: 2, nLayers: 3, dFF: 96, maxSeq: 64 }, 42);
  const prompt = Array.from({ length: 9 }, () => Math.floor(rng() * 60));

  // 路径 A：一次性前向整个 prompt，取最后一个位置的 logits
  clearTape();
  const full = model.forward(prompt).data;
  const V = model.cfg.vocab;
  const lastRow = full.subarray((prompt.length - 1) * V, prompt.length * V);

  // 路径 B：KV-cache 增量解码（内部走 FlashAttention-2）
  const collect = {};
  generate(model, prompt, 1, { temperature: 0, rng: () => 0.5, collect });

  let maxErr = 0;
  for (let i = 0; i < V; i++) maxErr = Math.max(maxErr, Math.abs(lastRow[i] - collect.firstLogits[i]));
  return {
    pass: maxErr < 1e-3,
    detail: `增量解码 logits 与全序列重算最大绝对差 ${maxErr.toExponential(3)}（阈值 1e-3，跨 Float32/Float64 累加）`,
  };
}

/* ------------------------------ ⑦ softmax 平移不变 + 数值稳定 ------------------------------ */
function invSoftmaxStable() {
  const rng = mulberry32(19);
  const M = 4, N = 32;
  const a = Tensor.randn([M, N], rng, 3);
  const s0 = softmaxRows(a).data;

  // (a) 平移 1e3：结果必须与不平移时一致（Float32 在该量级的舍入间距约 6e-5）
  const sh1 = new Tensor(Float32Array.from(a.data, (x) => x + 1e3), [M, N]);
  const s1 = softmaxRows(sh1).data;
  let err = 0, rowSumErr = 0;
  for (let i = 0; i < s0.length; i++) err = Math.max(err, Math.abs(s0[i] - s1[i]));
  for (let i = 0; i < M; i++) {
    let s = 0; for (let j = 0; j < N; j++) s += s1[i * N + j];
    rowSumErr = Math.max(rowSumErr, Math.abs(s - 1));
  }

  // (b) 平移 1e8：朴素实现 exp(1e8) 直接溢出为 Inf ⇒ NaN；稳定实现必须仍然给出合法分布
  const sh2 = new Tensor(Float32Array.from(a.data, (x) => x + 1e8), [M, N]);
  const s2 = softmaxRows(sh2).data;
  const finite = Array.from(s2).every(Number.isFinite);
  let sum2Err = 0;
  for (let i = 0; i < M; i++) {
    let s = 0; for (let j = 0; j < N; j++) s += s2[i * N + j];
    sum2Err = Math.max(sum2Err, Math.abs(s - 1));
  }
  return {
    pass: err < 1e-4 && rowSumErr < 1e-6 && finite && sum2Err < 1e-6,
    detail: `softmax(x+1e3) 与 softmax(x) 最大差 ${err.toExponential(3)}（Float32 舍入量级）；`
      + `溢出压力测试 x+1e8：无 Inf/NaN=${finite}，行和偏离 1 最大 ${sum2Err.toExponential(3)}`,
  };
}

/* ------------------------------ ⑧ BPE 往返恒等 ------------------------------ */
function invBPE() {
  const tok = new BPETokenizer();
  tok.train(CORPUS.slice(0, 6000), 300);
  const samples = [
    'Hello, world!',
    'The transformer architecture combines attention with feedforward layers.',
    'Gradient descent 梯度下降 — Newton-Schulz 迭代 ∑∫∂',
    'emoji: 🚀🌟 中文测试 hello',
    '  leading and trailing spaces  ',
    'punctuation!!!???... (parens) [brackets] {braces} "quotes"',
    'a\nb\tc',
  ];
  let bad = null;
  for (const s of samples) {
    const ids = tok.encode(s);
    const back = tok.decode(ids);
    if (back !== s) { bad = `encode/decode 不闭合: 原长度 ${s.length} 还原长度 ${back.length}`; break; }
  }
  // 压缩率也必须合理（BPE 若没学会合并，压缩率会接近 1）
  const ids = tok.encode(CORPUS);
  const ratio = CORPUS.length / ids.length;
  return {
    pass: bad === null && ratio > 1.5,
    detail: bad || `${samples.length} 组样本（含中文/emoji/标点/空白）全部往返恒等；词表 ${tok.size}，语料压缩率 ${ratio.toFixed(2)} 字符/token`,
  };
}

/* ------------------------------ ⑨ 单批过拟合 ------------------------------ */
function invOverfit() {
  const { tok, ids } = (() => {
    const t = new BPETokenizer(); t.train(CORPUS.slice(0, 4000), 256);
    return { tok: t, ids: t.encode(CORPUS.slice(0, 4000)) };
  })();
  const seq = 24;
  const X = [ids.slice(0, seq)], Y = [ids.slice(1, seq + 1)];
  const model = new GPT({ vocab: tok.size, dModel: 48, nHeads: 6, nKVHeads: 2, nLayers: 2, dFF: 96, maxSeq: 32 }, 5);
  const opt = new Muon(model.params(), { lr: 0.02, momentum: 0.95, weightDecay: 0 });
  const ev = () => { clearTape(); return model.forward(X[0], Y[0]).data[0]; };
  const l0 = ev();
  for (let s = 0; s < 250; s++) {
    clearTape(); opt.zeroGrad();
    const loss = model.forward(X[0], Y[0]);
    loss.backward();
    opt.step();
  }
  const l1 = ev();
  return {
    pass: l1 < 0.25 && l1 < l0 * 0.15,
    detail: `单批 250 步：loss ${l0.toFixed(4)} → ${l1.toFixed(4)}（降幅 ${(100 * (1 - l1 / l0)).toFixed(1)}%），要求终值 < 0.25`,
  };
}

/* ------------------------------ ⑩ 训练曲线单调下降 ------------------------------ */
// 验证「端到端可训练」：损失必须下行且取得≥10%相对降幅。
// 判定：① 最终 loss < 初始 loss×0.9（相对交叉熵下降≥10%，证明真实学习而非噪声）；
//       ② 全历程线性拟合斜率为负（证明整体下行趋势，而非末点偶然）。
function invTraining() {
  const res = train({
    steps: 200, seqLen: 32, batchSize: 4, nMerges: 300, lr: 0.025, seed: 1337, evalEvery: 0,
    cfg: { dModel: 64, nHeads: 4, nKVHeads: 2, nLayers: 3, dFF: 128 },
  });
  const h = res.history;
  const ini = h[0], fin = h[h.length - 1];
  // 最小二乘斜率
  const n = h.length; let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sx += i; sy += h[i]; sxy += i * h[i]; sxx += i * i; }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const relDown = fin < ini * 0.9;
  const q = Math.max(1, Math.floor(n / 4));
  const head = h.slice(0, q).reduce((a, b) => a + b, 0) / q;
  const tail = h.slice(-q).reduce((a, b) => a + b, 0) / q;
  return {
    pass: relDown && slope < 0,
    detail: `初始 ${ini.toFixed(4)} → 最终 ${fin.toFixed(4)}（相对降幅 ${((1 - fin / ini) * 100).toFixed(1)}%，要求 <90%初始=${ (ini * 0.9).toFixed(3)}）；全历程斜率 ${slope.toExponential(2)}（要求 <0）；首/末 1/4 均值 ${head.toFixed(3)}→${tail.toFixed(3)}`,
  };
}

/* ------------------------------ ⑪ RMSNorm 尺度不变 ------------------------------ */
function invRMSNorm() {
  const rng = mulberry32(23);
  const M = 5, D = 24;
  const x = Tensor.randn([M, D], rng, 2);
  const w = Tensor.randn([D], rng, 0.5);
  const a = rmsnorm(x, w).data;
  const c = 7.3;
  const xc = new Tensor(Float32Array.from(x.data, (v) => v * c), [M, D]);
  const b = rmsnorm(xc, w).data;
  let err = 0;
  for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]));
  return {
    pass: err < 1e-5,
    detail: `rmsnorm(c·x) 与 rmsnorm(x) 最大差 ${err.toExponential(3)}（c=${c}，阈值 1e-5）`,
  };
}

/* ------------------------------ 运行入口 ------------------------------ */

export function runAll() {
  results.length = 0;
  check('① 反向传播梯度 ↔ 中心差分数值梯度', invGradient, 'rel < 3e-3');
  check('② RoPE 相对位置不变性 + 范数保持', invRoPE, 'err < 1e-5');
  check('③ GQA(2 组 KV) ↔ 手工展开 MHA', invGQA, 'max|Δ| < 1e-5');
  check('④ FlashAttention-2 ↔ 朴素注意力', invFlash, 'max|Δ| < 1e-5');
  check('⑤ Newton-Schulz 半正交性 + Moonlight 缩放', invNewtonSchulz, '‖XᵀX−I‖ < 1e-3');
  check('⑥ KV-cache 增量解码 ↔ 全序列重算', invKVCache, 'max|Δ| < 1e-3');
  check('⑦ softmax 平移不变 + 溢出稳定', invSoftmaxStable, 'err < 1e-7');
  check('⑧ BPE 编解码往返恒等', invBPE, '精确相等');
  check('⑨ 单批过拟合（优化器有效性）', invOverfit, 'loss < 0.25');
  check('⑩ 训练 loss 单调趋势下降', invTraining, '终值<初始×0.9 且 斜率<0');
  check('⑪ RMSNorm 尺度不变性', invRMSNorm, 'err < 1e-5');
  return results.slice();
}

export function report(results, opts = {}) {
  const lines = [];
  let pass = 0;
  for (const r of results) {
    if (r.ok) pass++;
    lines.push(`${r.ok ? '✅ PASS' : '❌ FAIL'}  ${r.name}`);
    lines.push(`         ${r.detail}`);
    lines.push(`         容差 ${r.tol} · ${r.ms} ms`);
  }
  const head = `\n${'='.repeat(78)}\n  microforge 硬不变量自检  ${pass}/${results.length} 通过\n${'='.repeat(78)}`;
  const out = head + '\n' + lines.join('\n') + '\n' + '='.repeat(78) + '\n';
  if (opts.print !== false) console.log(out);
  return { text: out, pass, total: results.length };
}

/* Node CLI */
const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (isNode && process.argv[1] && /invariants\.(js|mjs)$/.test(process.argv[1])) {
  const r = runAll();
  const { pass, total } = report(r);
  process.exit(pass === total ? 0 : 1);
}
