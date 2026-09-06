# LOB Transformer

从零手写一个可运行的 Decoder-only Transformer，用于理解 LLM 的核心计算链路。

训练、反向、checkpoint、贪心生成与 KV Cache 如何接成一条链，见
[docs/训练与推理全链路.md](docs/训练与推理全链路.md)。下面是可执行命令和网页操作。

## 当前阶段

只依赖 NumPy 与标准库：字符级 Tokenizer、Embedding、RoPE、Causal Self-Attention、
Transformer Block、交叉熵、手写反向、SGD / Adam、checkpoint、贪心生成和请求内 KV Cache。
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

语料训练先按位置切分：前 90% 用于训练，末尾 10% 用于验证（至少 2 字符）；
可用 `--validation-fraction 0.2` 调整比例。两边至少各有 2 字符，因此文件至少需 4 字符。
窗口分别采样，不跨切分边界；每侧短文本自动缩短窗口。词表仍由整个语料构建，
验证文本不参与梯度更新，但字符集合已知；重复语料仍可能在两侧出现相同内容。

第 0 步、每 10 步和最后一步输出 `train_loss` / `val_loss`，分别来自两侧固定的
`batch-size` 个窗口，不是全语料平均值。独立随机数生成器确保评估不改变训练采样。
训练结束恢复所有已评估步骤中验证 loss 最低的权重（包括第 0 步），`--save` 保存此模型。
`best_step` / `best_val_loss` 标明选择结果；最后一步 loss 不一定是已保存模型的 loss。
CLI 和网页展示相同提示词在随机初始化与最佳模型上的续写，方便直观对比。
验证集用于选模型，不是独立测试集；小语料指标波动较大。

网页训练支持 4～100 万字符、最多 8191 种不同字符，请求体上限 16 MiB。
默认采用相同 10% 划分，显示训练/验证双曲线，完成后可启用最佳模型。
`--text` 保留单序列拟合，不划分验证集，保存最后一步权重。
文本在内存中读取，当前适合小型学习语料。扩大词表不会自动获得编程或对话能力。

未做：BPE、采样解码、流式输出。生产推理引擎留给 `lob-inference`。

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

## 已训练的小功能：中文时间转换

模型及配对语料位于 `data/time-task/`，所有操作均可在网页完成：

1. 在“模型训练”选择“中文时间转换 · 配对训练”。
2. 点击“生成示例语料”，或分别上传 `train.jsonl`、`validation.jsonl`、`test.jsonl`；也可导入包含这三份文件的文件夹。模型和报告不会混入训练语料。
3. 保持默认配置并开始训练，页面显示损失、验证准确率，支持停止任务。
4. 完成后查看测试准确率与失败案例，点击“启用模型并测试”。
5. 直接输入“下午三点半”，预期得到 `15:30`；网页自动处理等号和输出长度。

“模型管理”支持切换已保存模型、上传 NPZ、下载模型与三份语料、查看报告及重新评估原测试集。
上传 NPZ 时需选择其原本的功能；单独的模型文件不包含测试集和报告。
网页训练产物保存到 `web-checkpoints/` 的独立任务目录，刷新页面不会中断任务；服务重启后可在模型管理中找回已保存模型。

范围：上午一至十一点、中午十二点、下午一至六点、晚上七至十一点；
分钟支持零至五十九分，以及省略分钟、整、半。不支持日期、凌晨、两点或模糊时间。
输入须使用示例中的中文数字。该范围有意保持固定，未支持的表达会报错。

使用现有 NumPy Transformer（2 层、64 维、4 头），答案由模型逐字符生成。
输入格式为 `中文时间=HH:MM`，只对答案的五个字符计算损失；使用 Adam、梯度裁剪和 16 条样例的小批量。
按完整小时/分钟组合以 80%/10%/10% 划分，同一时间的别名不会分散到不同集合。
训练 1163 条、验证 143 条、测试 143 条；在第 300 步达到验证集 100%，
独立测试 141/143（98.60%），超过 95% 验收目标。
测试集只在根据验证集选定权重后评估，没有用测试结果继续调参。
这是固定语法、固定划分上的结果，不代表任意自然语言准确率。

两条测试失败保留在 `data/time-task/report.json`：
`下午三点` 预测为 `13:00`（正确 `15:00`）；
`晚上十一点` 原始预测为 `2:3:0`（正确 `23:00`，转换入口会拒绝此格式）。
格式校验不能发现第一种语义错误，当前模型不保证每条输入正确。

网页训练每 100 步评估验证集，保存准确率最高的权重；验证准确率达到 99% 时提前结束。
每次训练创建新目录，不覆盖示例模型。训练算法不增加 NumPy 之外的依赖。

## KV Cache 与网页测速

“模型测试”右侧新增 KV Cache 开关，网页默认开启，普通续写与时间转换均支持。
输入提示词后可点击“比较缓存开关速度”，在同一模型、提示词和生成长度下比较输出是否一致、
耗时、字符/秒和速度比。时间模型固定输出 5 个字符；普通模型使用页面设置的生成长度。
每种方式各预热一次，交替运行三次取耗时中位数；计时包含提示词处理，不包含网络传输。
短输出可能没有加速，训练期间的资源竞争也会影响测量。

缓存保存在每次生成请求内部，不跨请求或模型切换共享，不写入 checkpoint。
首次处理完整提示词并缓存每层 K/V，后续只处理新增字符，RoPE 使用对应位置偏移。
在上下文容量内，新旧推理路径数值近似一致；贪心输出已验证一致，但浮点运算差异在极端近似并列时可能影响选择。

窗口满后重建整个滑动窗口的缓存，以匹配原实现丢弃旧字符、重置位置并重新计算的行为。
不能只删除缓存开头的 K/V：高层历史状态已经包含被删除字符的信息。
因此超过窗口后每步都要重建，测速会显示重建次数；较长生成未必持续加速。
当前没有改变窗口语义，也不宣称实现了无限上下文。

验证包括多层模型的分段输入 logits、144 组随机生成、已有时间模型与跨窗口输出，
以及 HTTP 缓存开关、错误输入和测速接口。一次本地 2 层/64 维、上下文 128 的测量中，
16 字符提示词续写 64 字符：关闭约 24.7 ms，开启约 9.6 ms（约 2.6 倍）；实际以页面测量为准。
