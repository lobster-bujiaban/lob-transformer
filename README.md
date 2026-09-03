# LOB Transformer

从零手写一个可运行的 Decoder-only Transformer，用于理解 LLM 的核心计算链路。

## 当前阶段

第一阶段只依赖 NumPy，实现字符级 Tokenizer、Embedding、RoPE、Causal Self-Attention、
Transformer Block、Logits 和贪心生成。现已补齐交叉熵损失、手写反向传播与最小 SGD 训练。
`forward` / `generate` 仍使用随机初始化权重；`train` 在本次进程中训练后生成。

```bash
uv run lob-transformer --help
uv run lob-transformer tokenize --text "你好，Transformer"
uv run lob-transformer embedding --text "你好" --dimensions 8
uv run lob-transformer rope --text "你好" --dimensions 8
uv run lob-transformer attention --text "你好啊" --dimensions 8 --heads 2
uv run lob-transformer forward --text "你好，Transformer"
uv run lob-transformer generate --prompt "从前有座山" --tokens 32
uv run lob-transformer train --text "你好世界你好世界你好世界" --steps 200 --tokens 12
```

训练入口使用 `text[:-1]` 预测 `text[1:]`，计算平均交叉熵，沿 LM Head、
LayerNorm、MLP、Attention、RoPE 反传到 Embedding，更新全部参数。
使用全局梯度裁剪（阈值 1）和 SGD；LM Head 与 Embedding 独立更新，不共享权重。
当前只支持 2～129 字符的单条训练文本，不做静默截断；输出初始/最终损失，
并以首字符为提示生成。权重暂不保存，单条文本拟合不代表泛化能力。

后续阶段：权重保存与加载、数据批处理、Adam 优化器、BPE Tokenizer、KV Cache 和评测。

## 主链路

```text
文本 → Tokenizer → Embedding → Transformer Block × N → LayerNorm → LM Head → Logits → 贪心选择
                            每个 Block：
                            x = x + Attention(LayerNorm(x))
                            x = x + MLP(LayerNorm(x))
```

Attention 内部先投影 Q/K/V，再对 Q/K 应用 RoPE，通过因果掩码屏蔽未来位置。
LayerNorm 对每个 token 的特征维归一化，包含可学习的缩放和偏置；MLP 使用
`dimensions → dimensions × 4 → dimensions` 的无偏置线性层和 ReLU 激活。
各组件独立位于 `attention.py`、`normalization.py`、`mlp.py` 和 `block.py`。

本项目是学习型小模型，不追求 ChatGPT 级别能力；生产推理引擎留给 `lob-inference`。
