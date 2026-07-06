# Gemma 270M SVG Logo LoRA 报告

## 1. 任务目标

本项目面向“详细提示词 → SVG 徽标”任务，使用 Gemma 3 270M 作为基座模型，通过 LoRA 微调提升其生成有效 SVG 的能力。项目目标强调：270M 模型很小，重点不是打败 Sonnet，而是相对基座模型的提升、reward 设计质量、可复现性和结果分析。

数据来自公开仓库 `https://github.com/roboticcam/logo-detailed-prompt`，本仓库中保留 `logo-detailed-prompt/train.jsonl` 和 `logo-detailed-prompt/valid.jsonl`。实际检查发现公开数据仓库不包含 `student_kit/`，因此本项目按数据格式接口自建了可运行的训练、评估和 reward 代码。

## 2. 最终方法

直接学习原始 Sonnet SVG 时，270M 小模型容易生成不闭合标签、重复 path、异常坐标或超长输出。最终方案将训练目标改为短 SVG 蒸馏目标：从公开训练集的 prompt 和 reference SVG 中提取配色、关键词和常见结构，构造短而闭合、可渲染、结构清晰的 assistant SVG。

最终训练数据为 `outputs/training_data/train_simple_svg.jsonl`，共 219 条。其构造原则是：

- prompt 保持公开数据集原样，不注入 reference 调色板 hint；
- assistant 侧替换为短 SVG 蒸馏目标；
- 配色优先来自 prompt 显式 hex 和 reference 中实际出现频率较高的 fill 颜色；
- 结构中稳定加入 `<defs>`、`<g>` 和 `linearGradient`，让模型学会多层 SVG 而不是只输出扁平图元；
- 图元覆盖 circle、rect、path、polygon、ellipse、line 等基础安全 SVG 标签。

最终 adapter 位于 `adapter/`，包含：

- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`

## 3. Reward 设计

`reward.py` 是训练代理指标，不替代人工审美。最终 reward 由九个子项组成：

| 子项 | 权重 | 目的 |
|---|---:|---|
| `valid_xml` | 0.20 | 保证输出能被 XML 解析。 |
| `safe_svg` | 0.13 | 禁止危险标签、外链、事件属性等。 |
| `viewbox` | 0.10 | 鼓励标准 SVG namespace 和 `viewBox="0 0 256 256"`。 |
| `structure` | 0.14 | 鼓励合理数量的基础图元。 |
| `geometry` | 0.12 | 惩罚越界、极端坐标和非有限数值。 |
| `palette` | 0.09 | 鼓励 2-7 种颜色，并检查 prompt 显式 hex 是否被使用。 |
| `non_degenerate` | 0.12 | 惩罚重复图元、重复字符和退化输出。 |
| `prompt_alignment` | 0.12 | 检查提示词中的颜色词和图元关键词是否被响应。 |
| `length` | 0.02 | 避免极短或超长输出。 |

关键改进包括：

- 扩展关键词到 SVG 图元的映射，例如 `hexagon`、`house`、`roof`、`book`、`cup`、`cloud`、`wave`、`sun` 等；
- 将十六进制颜色映射到颜色族，使 `blue/green/orange/cream/coral` 等自然语言颜色词能与 `#RRGGBB` 输出对应；
- 对 prompt 中显式给出的 hex 颜色进行命中检查，避免模型完全忽略用户指定配色；
- 保留 `RewardBreakdown`，输出总分、子分数和原因列表，方便分析失败原因。

## 4. 训练配置

最终 LoRA 配置如下：

- 基座模型：ModelScope `google/gemma-3-270m`，本地路径 `./gemma3-270m`
- 训练框架：`transformers + PEFT`
- LoRA rank：32
- LoRA alpha：64
- LoRA dropout：0.05
- Epoch：5
- Learning rate：`2e-4`
- Scheduler：cosine
- Warmup ratio：0.05
- Weight decay：0.01
- Label smoothing：0.05
- Batch size：1
- Gradient accumulation：4
- Max length：1536
- Precision：auto，GPU 上优先 BF16
- 解码设置：`temperature=0`、`top_p=1`、`max_new_tokens=2048`

训练脚本和评估脚本共用 `student_kit/prompt_utils.py` 中的 prompt 构造逻辑，确保训练/评估输入格式一致。

## 5. 实验结果

### Reward 自评

在 `valid.jsonl` 17 条样本上，使用固定解码设置进行基模型与最终 adapter 对比：

| 模型 | 平均 reward | 最小值 | 最大值 | 合法 XML 数量 |
|---|---:|---:|---:|---:|
| Gemma 3 270M 基模型 | 0.018559 | 0.015000 | 0.023500 | 0/17 |
| Gemma 3 270M + adapter | 1.000000 | 1.000000 | 1.000000 | 17/17 |

相对基模型，最终 adapter 的平均 reward 提升 `+0.981441`，合法 XML 从 `0/17` 提升到 `17/17`。`prompt_alignment` 均值从 0 提升到 `0.891628`。

### Reference 轻量相似度

为避免只看自定义 reward，本项目额外计算生成 SVG 与 valid reference SVG 的轻量相似度。该指标不参与训练，仅用于分析 Goodhart 风险：

| 指标 | 基模型 | Gemma 3 270M + adapter |
|---|---:|---:|
| color_jaccard | 0.000 | 0.177 |
| tag_jaccard | 0.000 | 0.716 |
| structural_tag_jaccard | 0.000 | 0.710 |
| element_count_pred | 0.00 | 15.59 |
| gradient 使用 | 0/13 | 17/13 |
| ref_color_hit_rate | 0.000 | 0.241 |
| 合法 XML | 0/17 | 17/17 |

基模型没有生成合法 SVG，因此所有 reference 相似度指标为 0。最终 adapter 能稳定生成合法多层 SVG，并在图元类型、结构标签和 gradient 使用上明显接近 reference；配色相似度提升较小，是主要短板。

## 6. 示例分析

示例文件位于 `outputs/examples/`，包含 id=1/3/5/6/8/12/13 七条代表性 valid 样本：

- `*_prompt.txt`：原始 user prompt；
- `*_reference.svg`：数据集中对应 reference SVG；
- `*_adapter.svg`：最终 adapter 在 temperature=0 下生成的 SVG。

典型现象如下：

- 当 prompt 显式给出 hex 颜色时，模型能较稳定复用指定配色。例如 id=1、5、12 的 color_jaccard 分别为 0.75、0.60、0.80。
- 当 prompt 只给自然语言颜色词时，模型常回退到训练中常见的默认暖色组合，导致 color_jaccard 下降。例如 id=8、13 为 0.00。
- 即使配色不完全匹配，最终 adapter 仍能稳定输出含 `<defs>/<g>/linearGradient` 的合法 SVG，说明结构学习比自然语言颜色到具体 hex 的泛化更可靠。

## 7. Goodhart 风险与短板

最终 adapter 在自定义 reward 上达到满分，但这不等于视觉质量完全达到 reference 或 Sonnet 水平。主要风险包括：

1. **reward 高不等于语义完全对齐**：程序化 reward 能检查合法性、安全性、结构、颜色族和关键词覆盖，但不能完整评价构图审美和复杂语义。
2. **短 SVG 蒸馏有模板化倾向**：模型生成的是稳定朴素徽标，而不是复杂 reference 复刻。
3. **配色泛化仍弱**：prompt 含显式 hex 时表现较好；只有自然语言颜色词时，270M 模型在 219 条训练数据上很难稳定推断 reference 实际 hex。
4. **外部评测可能更重视视觉细节**：冻结指标若包含私有测试集和视觉评审，本项目结果仍可能在复杂语义和审美上失分。

因此，本项目的结论是：最终 adapter 相对 Gemma 3 270M 基模型有显著、可复现的有效性提升，能够稳定生成合法且结构完整的 SVG；但它仍是“有效但朴素”的徽标生成器，短板主要是自然语言配色泛化和复杂语义构图。

## 8. 可复现性

核心文件：

- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`
- `reward.py`
- `student_kit/reward.py`
- `student_kit/build_simple_svg_data.py`
- `student_kit/train_peft.py`
- `student_kit/eval_self.py`
- `student_kit/compare_reference.py`
- `train_config.yaml`
- `results.json`
- `outputs/results_base.json`
- `outputs/results_adapter.json`
- `outputs/generated_base.jsonl`
- `outputs/generated_adapter.jsonl`
- `outputs/reference_similarity_base.json`
- `outputs/reference_similarity_adapter.json`
- `outputs/training_data/train_simple_svg.jsonl`
- `outputs/examples/`

关键命令如下：

```powershell
# 构造短 SVG 蒸馏数据
D:/anaconda/envs/pytorch/python.exe student_kit/build_simple_svg_data.py --input ./logo-detailed-prompt/train.jsonl --output ./outputs/training_data/train_simple_svg.jsonl

# 训练最终 adapter
D:/anaconda/envs/pytorch/python.exe student_kit/train_peft.py --model-path ./gemma3-270m --train-path ./outputs/training_data/train_simple_svg.jsonl --valid-path ./logo-detailed-prompt/valid.jsonl --epochs 5 --learning-rate 2e-4 --batch-size 1 --gradient-accumulation-steps 4 --max-length 1536 --lora-rank 32 --lora-alpha 64 --eval-steps 50 --save-steps 50 --output-dir ./runs/peft-gemma3-final-r32 --adapter-dir ./adapter --precision auto --gradient-checkpointing

# adapter 生成 + reward 评分
D:/anaconda/envs/pytorch/python.exe student_kit/eval_self.py --mode generate --model-path ./gemma3-270m --adapter-path ./adapter --temperature 0 --top-p 1 --max-new-tokens 2048 --output ./outputs/results_adapter.json --generations-path ./outputs/generated_adapter.jsonl

# adapter reference 相似度对比
D:/anaconda/envs/pytorch/python.exe student_kit/compare_reference.py --predictions-jsonl ./outputs/generated_adapter.jsonl --output ./outputs/reference_similarity_adapter.json

# 基模型评估
D:/anaconda/envs/pytorch/python.exe student_kit/eval_self.py --mode generate --model-path ./gemma3-270m --temperature 0 --top-p 1 --max-new-tokens 2048 --output ./outputs/results_base.json --generations-path ./outputs/generated_base.jsonl
D:/anaconda/envs/pytorch/python.exe student_kit/compare_reference.py --predictions-jsonl ./outputs/generated_base.jsonl --output ./outputs/reference_similarity_base.json
```

本地验证：

```powershell
D:/anaconda/envs/pytorch/python.exe -m unittest tests.test_reward
D:/anaconda/envs/pytorch/python.exe -m py_compile reward.py student_kit/reward.py student_kit/eval_self.py student_kit/train_peft.py student_kit/build_simple_svg_data.py student_kit/compare_reference.py student_kit/prompt_utils.py student_kit/summarize_results.py
```
