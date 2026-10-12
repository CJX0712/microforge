#!/usr/bin/env node
/**
 * build.mjs — 把 src/*.js（纯 ESM，零依赖）内联进单文件 index.html
 *
 * 策略：剥离各文件的 import / export 行，按依赖顺序拼接进一个
 * 立即执行函数（IIFE），所有定义都在同一作用域，零运行时依赖、零网络请求。
 *
 * @author 晨星
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(__dirname, 'src');

// 依赖顺序：被依赖者在前
const ORDER = ['tensor.js', 'flash.js', 'bpe.js', 'muon.js', 'gpt.js', 'corpus.js', 'forge.js'];

/** 去掉 import / export 修饰符，保留实际定义（全部落入同一作用域） */
function strip(src) {
  const out = [];
  for (const raw of src.split('\n')) {
    const t = raw.trim();
    if (/^import\s/.test(t)) continue;                 // import 语句整行删除
    if (/^export\s*\{/.test(t)) continue;             // 分组 export 整行删除
    if (/^export\s+\*\s+as/.test(t)) continue;        // 命名空间重导出整行删除
    if (/^export\s+default/.test(t)) continue;        // 默认导出（本工程未使用）
    out.push(raw.replace(/^export\s+/, ''));
  }
  return out.join('\n');
}

let inner = '';
for (const f of ORDER) {
  inner += `\n/* ---- src/${f} ---- */\n`;
  inner += strip(fs.readFileSync(path.join(SRC, f), 'utf8')) + '\n';
}

// invariants.js：只去掉 import 行与 Node CLI 段，其余函数保留在 IIFE 内
let inv = fs.readFileSync(path.join(__dirname, 'tests', 'invariants.js'), 'utf8');
inv = inv.replace(/import\s*\{[\s\S]*?\}\s*from\s*['"][^'"]+[''];?/g, '');
inv = inv.replace(/\/\* ---- 运行入口 ----[\s\S]*$/, '');          // 去掉 Node CLI
inv = inv.replace(/function runAll\(\)\s*\{/, 'function runInvariants() {');  // 重命名

const bundle = `
const Forge = (() => {
const NS = {};
${inner}
/* ---- tests/invariants.js ---- */
${inv}
/* ---- 导出入口 ---- */
NS.GPT = GPT; NS.generate = generate; NS.Muon = Muon; NS.BPETokenizer = BPETokenizer;
NS.train = train; NS.CORPUS = CORPUS; NS.DEFAULT_CONFIG = DEFAULT_CONFIG;
NS.moonlightScale = moonlightScale; NS.newtonSchulz = newtonSchulz;
NS.attentionFlash = attentionFlash; NS.attentionNaive = attentionNaive;
NS.clearTape = clearTape; NS.mulberry32 = mulberry32; NS.Tensor = Tensor;
NS.runInvariants = runInvariants; NS.report = report;
NS.VERSION = VERSION; NS.AUTHOR = AUTHOR;
NS.T = { Tensor, rope, softmaxRows, rmsnorm, mul, transpose, matmul, reshape, splitHeads, mergeHeads, sliceLast, setNoGrad };
return NS;
})();
`;

const html = fs.readFileSync(path.join(__dirname, 'demo', 'template.html'), 'utf8');
const out = html.replace('/*__BUNDLE__*/', bundle);
fs.writeFileSync(path.join(__dirname, 'index.html'), out);
console.log('✅ index.html built (' + (out.length / 1024).toFixed(1) + ' KB)');
