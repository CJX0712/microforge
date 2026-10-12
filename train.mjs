#!/usr/bin/env node
/**
 * train.mjs — Node 无头训练 + 采样生成
 *
 *   node train.mjs                 # 默认配置
 *   node train.mjs --steps 1200    # 自定义步数
 *   node train.mjs --text my.txt   # 自定义语料
 *
 * @author 晨星
 */

import fs from 'node:fs';
import { train, generate, corpusStats, VERSION, AUTHOR } from './src/forge.js';
import { CORPUS } from './src/corpus.js';

const argv = process.argv.slice(2);
const arg = (k, d) => {
  const i = argv.indexOf('--' + k);
  return i >= 0 ? argv[i + 1] : d;
};
const num = (k, d) => Number(arg(k, d));

const text = argv.includes('--text') ? fs.readFileSync(arg('text', ''), 'utf8') : CORPUS;
const steps = num('steps', 800);
const seqLen = num('seqLen', 64);
const batchSize = num('batchSize', 8);
const lr = num('lr', 0.02);
const nMerges = num('merges', 420);
const seed = num('seed', 1337);

console.log(`\nmicroforge v${VERSION}  ·  author: ${AUTHOR}`);
console.log('='.repeat(72));
const st = corpusStats(text);
console.log(`语料：${st.chars} 字符 / ${st.words} 词 / ${st.lines} 段`);
console.log(`配置：steps=${steps} seqLen=${seqLen} batch=${batchSize} lr=${lr} merges=${nMerges} seed=${seed}`);
console.log('-'.repeat(72));

const t0 = Date.now();
let last = t0;
const res = train({
  text, steps, seqLen, batchSize, lr, nMerges, seed,
  onEval: ({ step, trainLoss, valLoss }) => {
    const now = Date.now();
    const ppl = Math.exp(valLoss);
    console.log(
      `step ${String(step).padStart(4)} | train ${trainLoss.toFixed(4)} | val ${valLoss.toFixed(4)} `
      + `| val ppl ${ppl.toFixed(2).padStart(7)} | ${((now - last) / 1000).toFixed(1)}s`
    );
    last = now;
  },
});
const { model, tok, history } = res;

console.log('-'.repeat(72));
const head = history.slice(0, Math.floor(history.length / 4));
const tail = history.slice(-Math.floor(history.length / 4));
const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
console.log(`参数量：${model.paramCount().toLocaleString('en-US')}`);
console.log(`词表大小：${tok.size}`);
console.log(`loss：${avg(head).toFixed(4)} → ${avg(tail).toFixed(4)}   perplexity：${Math.exp(avg(head)).toFixed(1)} → ${Math.exp(avg(tail)).toFixed(1)}`);
console.log(`总耗时：${((Date.now() - t0) / 1000).toFixed(1)}s`);

console.log('\n采样生成（temperature 0.8 / top-k 40）：');
console.log('-'.repeat(72));
const prompts = ['Artificial intelligence', 'The transformer architecture', 'Mathematics is'];
for (const p of prompts) {
  const ids = tok.encode(p);
  if (ids.length === 0) continue;
  const outIds = generate(model, ids, 60, { temperature: 0.8, topK: 40 });
  console.log(`▶ ${p}\n  ${p}${tok.decode(outIds)}\n`);
}
console.log('='.repeat(72));
