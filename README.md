# LOB Transformer

从零手写一个可运行的 Decoder-only Transformer，用于理解 LLM 的核心计算链路。

## 当前阶段

第一阶段只依赖 NumPy，实现字符级 Tokenizer、Embedding、RoPE、Causal Self-Attention、
Transformer Block、Logits 和贪心生成。现已补齐交叉熵损失、手写反向传播与最小 SGD 训练。
`forward` / 未指定 checkpoint 的 `generate` 使用随机初始化权重；`train` 训练后可保存。

```bash
uv run lob-transformer --help
uv run lob-transformer tokenize --text "你好，Transformer"
uv run lob-transformer embedding --text "你好" --dimensions 8
uv run lob-transformer rope --text "你好" --dimensions 8
uv run lob-transformer attention --text "你好啊" --dimensions 8 --heads 2
uv run lob-transformer forward --text "你好，Transformer"
uv run lob-transformer generate --prompt "从前有座山" --tokens 32
uv run lob-transformer train --text "你好世界你好世界你好世界" --steps 200 --tokens 12
uv run lob-transformer train --text "你好世界你好世界你好世界" --steps 200 --save model.npz
uv run lob-transformer generate --checkpoint model.npz --prompt "你好" --tokens 12
```

训练入口使用 `text[:-1]` 预测 `text[1:]`，计算平均交叉熵，沿 LM Head、
LayerNorm、MLP、Attention、RoPE 反传到 Embedding，更新全部参数。
使用全局梯度裁剪（阈值 1）和 SGD；LM Head 与 Embedding 独立更新，不共享权重。
`--text` 默认支持 2～129 字符的单条训练文本，不做静默截断；输出初始/最终损失，
并以首字符为提示生成。单条文本拟合不代表泛化能力。

`--save` 将全部权重、模型配置和原始词表保存到一个 NPZ 文件（不使用 pickle）。
目标目录须已存在；同名文件会原子替换。`--checkpoint` 加载时校验版本、词表、
参数名称、形状及数值；生成使用保存的词表，不会根据提示词重新构建。
提示词包含训练词表之外的字符会报错。保存的是模型而非训练进度，不含优化器状态。

## 文本语料与小批量训练

```bash
uv run lob-transformer train --file data/example.txt --context-length 32 --batch-size 4 --steps 200 --save corpus.npz
uv run lob-transformer generate --checkpoint corpus.npz --prompt "你好" --tokens 32
./start.sh --checkpoint corpus.npz
```

将示例路径换成你的 UTF-8 文本即可。整个文件（包括换行）构建字符词表，至少需要两个字符。
每一步从全部合法起点有放回采样 `batch-size` 个连续窗口，窗口包含
`context-length + 1` 个字符，错开一位形成输入与目标；短文件自动缩短窗口，无需补齐。
每个窗口独立前向/反向，梯度取平均后裁剪并执行一次 SGD 更新。
这是梯度累积式小批量，不是向量化并行计算；`--steps` 是更新次数，不是遍历语料的轮数。
`--seed` 默认 7，控制初始化与采样。`--text` 保留单序列训练，`--batch-size` 仅用于 `--file`。

每 10 步和最后一步输出固定训练样本上的 `train_sample_loss`，初始/最终损失也基于相同样本，
不是独立验证集指标，也不是全语料平均值。文本在内存中读取，当前适合小型学习语料。
扩大词表只解决字符覆盖，不会自动获得编程或对话能力；训练新权重后需重启网页服务。

后续阶段：独立验证集、Adam 优化器、BPE Tokenizer、KV Cache 和评测。

## HTTP 推理接口

先用上面的训练命令保存 `model.npz`，再启动服务（启动时只加载一次）：

```bash
uv run lob-transformer serve --checkpoint model.npz --port 8000
```

也可一键启动（从任意目录调用均可，默认加载项目内的 `model.npz`）：

```bash
./start.sh
./start.sh --port 8012
./start.sh --checkpoint another-model.npz --port 8001
```

需要先安装 `uv` 并生成 checkpoint；按 Ctrl+C 停止服务。

浏览器打开 `http://127.0.0.1:8000/` 即可使用 Web 推理实验台：编辑提示词、
调整生成字符数、查看续写和请求耗时、复制结果。页面与接口同源，无需前端构建或 CDN。
页面参考 LOB Vector 系列的浅色绿色布局，并附模型链路说明和接口示例。

也可以在另一个终端调用：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"你好","tokens":12}'
```

生成响应示例：

```json
{"text":"你好世界你好世界你好世界你好","completion":"世界你好世界你好世界你好","prompt_tokens":2,"generated_tokens":12}
```

`text` 包含提示词，`completion` 仅包含新生成内容。`tokens` 默认 16，允许 0～256；
提示词须非空、不超过模型上下文长度，且字符均在保存的词表中。
请求体最多 16 KiB，读取超时 10 秒；错误返回 JSON `error` 和相应 HTTP 状态码。
默认监听 `127.0.0.1`，使用标准库串行处理请求，不新增依赖。
这是本地开发接口，不兼容 Ollama/OpenAI 协议，不支持流式输出、鉴权或 TLS；
不要直接暴露到公网。按 Ctrl+C 停止；更换 checkpoint 后需重启服务。

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
