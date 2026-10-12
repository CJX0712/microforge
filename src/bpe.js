/**
 * bpe.js — Byte-Level BPE 分词器（GPT-2 血统，纯 JS 零依赖）
 *
 * 血统：
 *   - Byte Pair Encoding 算法本体：Sennrich et al. 2016 (ACL) "Neural Machine Translation of Rare Words with Subword Units"
 *   - Byte-level 变体 + 可打印字节映射表：Radford et al. 2019 (GPT-2)，OpenAI tiktoken 沿用同一套
 *   - 预分词正则（'s / 't / 字母块 / 数字块 / 其他）：GPT-2 tokenizer.py 原样
 *
 * 不变量（tests/invariants.js 钉死）：∀ text, decode(encode(text)) === text
 *
 * @author 晨星
 */

/* ---- GPT-2 bytes→unicode 映射：256 个字节 ↔ 256 个连续码位，一一对应 ---- */
const BYTE_ENC = (() => {
  const arr = new Array(256);
  for (let b = 0; b < 256; b++) arr[b] = String.fromCodePoint(256 + b);
  return arr;
})();
const BYTE_DEC = (() => {
  const m = new Map();
  for (let b = 0; b < 256; b++) m.set(String.fromCodePoint(256 + b), b);
  return m;
})();

/** GPT-2 预分词正则 */
const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class BPETokenizer {
  constructor() {
    this.merges = new Map();   // "a|b" -> rank
    this.vocab = new Map();    // token string -> id
    this.idToToken = [];
    for (let b = 0; b < 256; b++) { this.vocab.set(BYTE_ENC[b], b); this.idToToken[b] = BYTE_ENC[b]; }
    this.nextId = 256;
  }

  /** 在语料上训练 N 次合并（经典 BPE：每次合并频次最高的一对） */
  train(text, nMerges = 400, verbose = false) {
    let words = this._preSplit(text);
    // words: Map<wordString, count>，word 内部为「以空格分隔的字节符号串」
    let freq = new Map();
    for (const w of words) freq.set(w, (freq.get(w) || 0) + 1);

    // 每个 word 表示为符号数组
    let corpus = new Map();
    for (const [w, c] of freq) corpus.set(w, { toks: w.split(' '), count: c });

    for (let it = 0; it < nMerges; it++) {
      const pairCount = new Map();
      for (const { toks, count } of corpus.values()) {
        for (let i = 0; i < toks.length - 1; i++) {
          const key = toks[i] + '|' + toks[i + 1];
          pairCount.set(key, (pairCount.get(key) || 0) + count);
        }
      }
      if (pairCount.size === 0) break;
      let best = null, bestC = -1;
      for (const [k, c] of pairCount) if (c > bestC || (c === bestC && k < best)) { best = k; bestC = c; }
      const [a, b] = best.split('|');
      const rank = this.merges.size;
      this.merges.set(best, rank);
      const merged = a + b;
      if (!this.vocab.has(merged)) { this.vocab.set(merged, this.nextId); this.idToToken[this.nextId] = merged; this.nextId++; }
      // 应用到语料
      const ncorpus = new Map();
      for (const [w, { toks, count }] of corpus) {
        const nt = [];
        for (let i = 0; i < toks.length;) {
          if (i < toks.length - 1 && toks[i] === a && toks[i + 1] === b) { nt.push(merged); i += 2; }
          else { nt.push(toks[i]); i++; }
        }
        ncorpus.set(nt.join(' '), { toks: nt, count });
      }
      corpus = ncorpus;
      if (verbose && it % 50 === 0) process.stdout?.write?.(`\r  BPE merge ${it}/${nMerges}  best="${a}+${b}" ×${bestC}`);
    }
    if (verbose) console.log('');
    return this;
  }

  _preSplit(text) {
    const out = [];
    const matches = text.match(PAT) || [];
    for (const m of matches) {
      let s = '';
      // 按码位迭代（Array.from 正确处理代理对 ⇒ emoji / 非 BMP 汉字不丢）
      for (const ch of Array.from(m)) {
        const code = ch.codePointAt(0);
        if (code < 0x80) s += BYTE_ENC[code];
        else if (code < 0x800) s += BYTE_ENC[0xC0 | (code >> 6)] + BYTE_ENC[0x80 | (code & 0x3F)];
        else if (code < 0x10000) s += BYTE_ENC[0xE0 | (code >> 12)] + BYTE_ENC[0x80 | ((code >> 6) & 0x3F)] + BYTE_ENC[0x80 | (code & 0x3F)];
        else s += BYTE_ENC[0xF0 | (code >> 18)] + BYTE_ENC[0x80 | ((code >> 12) & 0x3F)] + BYTE_ENC[0x80 | ((code >> 6) & 0x3F)] + BYTE_ENC[0x80 | (code & 0x3F)];
      }
      out.push(Array.from(s).join(' '));   // 每个字节符号用空格分隔
    }
    return out;
  }

  /** 单个预分词块 → merge 后的符号数组 */
  _bpe(wordSyms) {
    let toks = wordSyms;
    if (toks.length < 2) return toks;
    while (true) {
      let bestRank = Infinity, bestIdx = -1;
      for (let i = 0; i < toks.length - 1; i++) {
        const r = this.merges.get(toks[i] + '|' + toks[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
      }
      if (bestIdx === -1) break;
      toks = toks.slice(0, bestIdx).concat([toks[bestIdx] + toks[bestIdx + 1]], toks.slice(bestIdx + 2));
    }
    return toks;
  }

  /** text → ids */
  encode(text) {
    const ids = [];
    for (const chunk of this._preSplit(text)) {
      for (const tok of this._bpe(chunk.split(' '))) {
        const id = this.vocab.get(tok);
        if (id === undefined) {   // 理论上不会发生：未知 token 退化为逐字节
          for (const c of tok.split('')) ids.push(this.vocab.get(c));
        } else ids.push(id);
      }
    }
    return ids;
  }

  /** ids → text */
  decode(ids) {
    let s = '';
    for (const id of ids) s += this.idToToken[id] || '';
    // 逆映射：unicode 码位 → UTF-8 字节
    const bytes = [];
    for (const ch of s) {
      const b = BYTE_DEC.get(ch);
      if (b !== undefined) bytes.push(b);
    }
    // 手写 UTF-8 解码，零依赖。
    // 关键：必须用 fromCodePoint —— fromCharCode 会把 >0xFFFF 的码位截断成代理对的低位，
    // 导致 emoji（4 字节序列）被还原成 U+F680 之类的错误字符。
    let out = '';
    let i = 0;
    while (i < bytes.length) {
      const b0 = bytes[i];
      if (b0 < 0x80) { out += String.fromCodePoint(b0); i += 1; }
      else if (b0 >= 0xC0 && b0 < 0xE0) { out += String.fromCodePoint(((b0 & 0x1F) << 6) | (bytes[i + 1] & 0x3F)); i += 2; }
      else if (b0 >= 0xE0 && b0 < 0xF0) { out += String.fromCodePoint(((b0 & 0x0F) << 12) | ((bytes[i + 1] & 0x3F) << 6) | (bytes[i + 2] & 0x3F)); i += 3; }
      else { out += String.fromCodePoint(((b0 & 0x07) << 18) | ((bytes[i + 1] & 0x3F) << 12) | ((bytes[i + 2] & 0x3F) << 6) | (bytes[i + 3] & 0x3F)); i += 4; }
    }
    return out;
  }

  get size() { return this.nextId; }
}

export { BYTE_ENC, BYTE_DEC };
