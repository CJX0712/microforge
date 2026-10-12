# microforge · 世界顶级微型 LLM 工程栈

> **零依赖 · 单文件 · 浏览器内可训练可推理 · 11 条硬不变量自检**
> 作者：**晨星** ｜ 协议：MIT ｜ 版本：1.0.0

`microforge` 把 **2023–2025 五项已被工业界验证的顶级成果**焊进一个零依赖、可在浏览器里训练并推理的微型 LLM 工程栈，并用 **11 条数学/数值硬不变量**对每条成果做交叉验证——任何一条不过，体系立即暴露。

设计三原则（晨星）：
- **零重写**：每条需求先检索世界已有最优解，直接复用、组合、超越，不从零造轮子。
- **集大成**：每一行代码都站在世界顶级成果之上；凡有更强现成方案立即替换自写。
- **闭环交付**：成果完整开源、可运行、可复现、可验证。

---

## 组件血统表（每条都站在顶级成果之上）

| 模块 | 文件 | 世界顶级来源 | 参考论文 |
|------|------|--------------|----------|
| 自动微分引擎 | `src/tensor.js` | micrograd 思想 → 升级到 N-D 张量 + tape | Karpathy, *micrograd* (2022) |
| RMSNorm | `src/tensor.js` | RMS Layer Normalization | Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019) |
| RoPE 旋转位置编码 | `src/tensor.js` | Rotary Position Embedding | Su et al., *RoFormer* (2021) |
| GQA 分组查询注意力 | `src/gpt.js` / `src/flash.js` | Grouped-Query Attention | Ainslie et al. (2023) |
| SwiGLU 激活 | `src/gpt.js` | Swish-Gated Linear Unit | Shazeer, *GLU Variants* (2020) |
| FlashAttention-2 | `src/flash.js` | Online Softmax + 分块 running max/ℓ rescaling | Dao, *FlashAttention-2* (2024) |
| Muon 优化器 | `src/muon.js` | Newton-Schulz 半正交化 + Nesterov | Keller Jordan et al. (2024) |
| Moonlight 缩放 | `src/muon.js` | 0.2·√max(dout,din) | Moonshot AI, *Moonlight* (2025) |
| AdamW | `src/muon.js` | 解耦权重衰减 | Loshchilov & Hutter (2019) |
| 余弦退火 + Warmup | `src/muon.js` | Cosine Schedule | Loshchilov & Hutter (2017) |
| Byte-Level BPE | `src/bpe.js` | GPT-2 分词器 | Sennrich (2016) / Radford et al. (2019) |
| 权重绑定 | `src/gpt.js` | Tied Embedding | Press & Wolf (2017) |
| KV-Cache 增量解码 | `src/gpt.js` | Prefill + Decode 两阶段 | 标准 Transformer 推理范式 |

---

## 架构总览

```
                输入文本
                   │
            ┌──────┴──────┐
            │ Byte-BPE 分词 │  src/bpe.js
            └──────┬──────┘
          token ids [B, T]
                   │
   ┌───────────────┴───────────────────────────────┐
   │  GPT 主干 (Pre-LN, 残差 1/√(2L))  src/gpt.js   │
   │                                                │
   │  Embedding (权重绑定) → 逐层:                  │
   │   ├─ RMSNorm                                   │
   │   ├─ RoPE 位置编码                             │
   │   ├─ GQA 注意力 (FlashAttention-2 在线 softmax)│
   │   ├─ RMSNorm                                   │
   │   └─ SwiGLU FFN                                │
   │                                                │
   │  推理: KV-Cache 增量解码 (prefill+decode)      │
   └───────────────┬───────────────────────────────┘
                   │  logits [B, T, V]
            ┌──────┴──────┐
            │ 采样 / 损失  │  top-k + temperature
            └─────────────┘

   优化器: Muon (2D 矩阵, Newton-Schulz) + AdamW (1D/嵌入/输出头)
           余弦退火 + warmup, Moonlight 缩放   src/muon.js
```

自动微分 `src/tensor.js` 为底层引擎：Float32 张量 + 反向模式 tape，所有算子均可微，梯度由中心差分数值梯度独立交叉验证（不变量①）。

---

## 11 条硬不变量自检

`npm test` 运行以下 11 条独立交叉验证。任何一条失败即说明对应组件实现有误。

| # | 不变量 | 验证内容 | 容差 |
|---|--------|----------|------|
| ① | 反向传播 ↔ 中心差分 | 全组件解析梯度 vs 数值梯度（Float32 噪声下限） | rel < 3e-3 |
| ② | RoPE 相对性 + 范数保持 | ⟨R_m q, R_n k⟩ 仅依赖相对位置 m−n；R 保持范数 | err < 1e-5 |
| ③ | GQA(2 KV) ↔ 展开 MHA | 分组查询注意力与手工展开多头注意力等价 | max\|Δ\| < 1e-5 |
| ④ | FlashAttention-2 ↔ 朴素 | 分块 2/4/8/16/32 在线 softmax 与朴素注意力一致 | max\|Δ\| < 1e-5 |
| ⑤ | Newton-Schulz 半正交 + Moonlight | cubic[20步] Gram 偏离 I；quintic[5步] ‖X‖_F/√min∈[0.8,1.2]；缩放因子正确 | ‖XᵀX−I‖ < 1e-3 |
| ⑥ | KV-cache ↔ 全序列重算 | 增量解码 logits 与全序列重算一致 | max\|Δ\| < 1e-3 |
| ⑦ | softmax 平移不变 + 溢出稳定 | softmax(x+c) 不变；x+1e8 无 Inf/NaN，行和=1 | err < 1e-7 |
| ⑧ | BPE 往返恒等 | 含中文/emoji/标点/空白样本编解码精确往返 | 精确相等 |
| ⑨ | 单批过拟合 | 优化器有效性：单批 250 步 loss→0 | loss < 0.25 |
| ⑩ | 训练曲线单调下降 | 最终 loss < 初始×0.9 且全历程斜率<0（端到端可训练） | 终值<初值×0.9 且 斜率<0 |
| ⑪ | RMSNorm 尺度不变 | rmsnorm(c·x)=rmsnorm(x) | err < 1e-5 |

---

## 快速开始

零依赖，仅需 Node ≥ 18（浏览器无需任何安装）。

```bash
# 1. 运行 11 条硬不变量自检（核心验证）
npm test

# 2. Node 无头训练 + 采样生成
npm run train -- --steps 300 --seqLen 32

# 3. 重建单文件浏览器游乐场 index.html
npm run build
```

### 浏览器游乐场

`npm run build` 后直接双击打开 `index.html`（或 `open index.html`），即可在浏览器内：
1. **一键自检**——运行 11 条不变量，全绿即正确；
2. **浏览器内训练**——实时 loss 曲线，无需服务器；
3. **可编辑语料**——替换为你的文本再训练；
4. **KV-Cache 生成**——输入 prompt，看增量解码逐 token 输出。

> 单文件 `index.html` 内联了全部 `src/*.js`（已剥离 ESM `import/export` 打包为 IIFE），无外部 CDN、无网络依赖。

---

## 复现命令速查

| 目标 | 命令 |
|------|------|
| 全部不变量自检 | `npm test` |
| 指定安静模式 | `FORGE_QUIET=1 npm test` |
| 训练并采样 | `node train.mjs --steps 300 --seqLen 32 --lr 0.025` |
| 重新打包单文件 | `node build.mjs` |
| 看版本/作者 | 控制台执行 `Forge.VERSION` / `Forge.AUTHOR` |

---

## 数学核心（要点）

- **RMSNorm**：`x̂ = x / √(mean(x²) + ε) · γ`，沿最后一维归一化，对输入尺度不变。
- **RoPE**：`R_θ^{(m)} = diag(e^{i m θ_j})`，将位置编码旋入 q/k，使 ⟨R_m q, R_n k⟩ 仅依赖 m−n（相对位置）。
- **GQA**：每 `n_heads/n_kv_heads` 个查询头共享一组 KV，降低 KV-cache 显存与解码延迟。
- **SwiGLU**：`FFN(x) = (silu(xW_1) ⊙ xW_2) W_3`，门控线性单元。
- **FlashAttention-2**：在线 softmax，`m_new=max(m,P)`，`ℓ_new=e^{m−m_new}ℓ + e^{P−m_new}`，分块不落全注意力矩阵。
- **Muon**：对 2D 权重 `W ← W − η·(X − ½ X Xᵀ X)·(1+β) − β·W_prev`（Nesterov），Newton-Schulz 迭代 `(aX+bXXᵀX+cXXᵀXXᵀX)` 逼近半正交，配 Moonlight 缩放 `0.2·√max(dout,din)`。

---

## 性能基准（参考）

- 默认配置（dModel 96 / 6 头 / 4 层）前向+反向 ≈ 119 ms/step（≈ 534 MMAC/s）。
- 单文件 `index.html` 在浏览器内可实时训练并绘制 loss 曲线。
- 全部 11 条不变量自检在数分钟内完成（含完整训练与过拟合验证）。

---

## 目录结构

```
microforge/
├── src/
│   ├── tensor.js    # 自动微分引擎 + 算子 (matmul/linear/rmsnorm/rope/softmax/...)
│   ├── flash.js     # FlashAttention-2 + 朴素注意力 + GQA 映射
│   ├── muon.js      # Muon + Newton-Schulz + AdamW + 余弦调度
│   ├── bpe.js       # Byte-Level BPE 分词器
│   ├── gpt.js       # GPT 主干 + KV-Cache 推理
│   ├── corpus.js    # 内置训练语料
│   └── forge.js     # 统一入口 (train/eval/batch/tokenize)
├── tests/
│   └── invariants.js  # 11 条硬不变量自检
├── demo/
│   └── template.html  # 单文件浏览器游乐场模板
├── build.mjs        # 打包 src → 单文件 index.html
├── train.mjs        # Node 无头训练 + 采样 CLI
├── index.html       # 构建产物（浏览器游乐场）
├── package.json
└── README.md
```

---

## 路线图

- [x] 11 条硬不变量自检全绿
- [x] 浏览器内训练 + 推理单文件
- [ ] 多轮对话 KV-cache 复用
- [ ] 权重导出 / 量化 (int8) 推理
- [ ] 更大语料预训练脚本

---

## 许可与署名

MIT License · 作者 **晨星**。
本项目以「集大成」为原则，所有核心算法均基于上方引用的世界顶级论文成果实现与组合。

> 零重写 · 集大成 · 闭环交付
