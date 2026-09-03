# LOB Transformer

从零手写一个可运行的 Decoder-only Transformer，用于理解 LLM 的核心计算链路。

## 当前阶段

第一阶段只依赖 NumPy，实现字符级 Tokenizer、Embedding、RoPE、Causal Self-Attention、
Transformer Block、Logits 和贪心生成。当前权重为随机初始化，重点是验证张量形状和前向链路。

```bash
uv run lob-transformer --help
uv run lob-transformer forward --text "你好，Transformer"
uv run lob-transformer generate --prompt "从前有座山" --tokens 32
```

后续阶段：反向传播与训练、BPE Tokenizer、优化器、数据批处理、KV Cache 和评测。

## 主链路

```text
文本 → Tokenizer → Embedding → RoPE → Attention → MLP → LM Head → Logits → Sampling
```

本项目是学习型小模型，不追求 ChatGPT 级别能力；生产推理引擎留给 `lob-inference`。
