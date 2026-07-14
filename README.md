# Gemma 270M SVG Logo LoRA

本仓库实现“详细提示词 → SVG 徽标”的小模型微调项目：自定义 `reward.py`、LoRA 微调 Gemma 3 270M、基模型/微调自评、reference 轻量相似度评估和报告分析。

## 当前状态

- 数据仓库：`logo-detailed-prompt/`，来自 `https://github.com/roboticcam/logo-detailed-prompt`。
- 说明：实际 GitHub 数据仓库只包含 `README.md`、`train.jsonl`、`valid.jsonl`，不包含 `student_kit/`；本项目按数据格式接口自建了可运行版本。
- 基座模型：ModelScope `google/gemma-3-270m`，已下载到 `gemma3-270m/`。
- 最终 LoRA adapter：`adapter/adapter_config.json` 与 `adapter/adapter_model.safetensors`（r32 + 短 SVG 蒸馏数据）。
- 基线对比采用基模型（Gemma 3 270M 不加载 adapter），自评产物在 `outputs/results_base.json` / `outputs/results_adapter.json`。
- 最终自评（valid 17 条，temperature 0）：基模型平均 reward `0.017845`、0/17 合法 XML；`adapter` 平均 reward `0.986724`、17/17 合法 XML、`prompt_alignment` 均值 `0.892`。
- Reference 轻量相似度：`adapter` 颜色 Jaccard `0.177`、tag Jaccard `0.716`、structural_tag Jaccard `0.710`、element_count_pred `15.59`；基模型全部为 0。详见 `report.md` 第 5 节。

## 主要文件

- `reward.py`：对外入口，转发到 `student_kit.reward`。
- `student_kit/reward.py`：可解释 SVG reward，含颜色族匹配和扩展图元关键词。
- `student_kit/build_simple_svg_data.py`：从公开训练集 prompt/reference 构造短 SVG 蒸馏目标（频次配色、去除衍生混色、加入 `<defs>/<g>/linearGradient` 结构）。
- `student_kit/prompt_utils.py`：训练/评估共用的 `build_prompt` 实现，保证 prompt 模板一致。
- `student_kit/train_peft.py`：Transformers + PEFT LoRA 训练脚本，支持 prompt mask、BF16、checkpoint resume。
- `student_kit/eval_self.py`：基模型/LoRA/参考 SVG 自评脚本。
- `student_kit/compare_reference.py`：生成 SVG 与 reference 的颜色/tag/结构标签 Jaccard 与 element_count 比值等扩展指标。
- `train_config.yaml`：最终训练、评估和实验配置。
- `results.json`：基模型 vs 最终 adapter 的验证集自评汇总与 reference 相似度汇总。
- `report.md`：优化思路、实验结果、案例和 Goodhart 分析。
- `outputs/examples/`：7 组（id=1/3/5/6/8/12/13）prompt / reference / adapter 对比样例。

## 环境

本机使用已有 Anaconda PyTorch 环境，未向系统 Python 安装训练依赖：

```powershell
D:/anaconda/envs/pytorch/python.exe
```

关键依赖：

- `torch 2.11.0+cu128`
- `transformers 5.13.0`
- `peft 0.19.1`
- `accelerate`
- `sentencepiece`
- `safetensors`
- `modelscope`

## 复现命令

仓库根目录执行。涉及 `student_kit` 包内导入的训练与评估脚本使用 `-m student_kit.<模块名>` 运行。

### 测试

```powershell
D:/anaconda/envs/pytorch/python.exe -m unittest tests.test_reward
D:/anaconda/envs/pytorch/python.exe -m py_compile reward.py student_kit/reward.py student_kit/eval_self.py student_kit/train_peft.py student_kit/build_simple_svg_data.py student_kit/compare_reference.py
```

### 构造短 SVG 蒸馏数据

```powershell
D:/anaconda/envs/pytorch/python.exe student_kit/build_simple_svg_data.py --input ./logo-detailed-prompt/train.jsonl --output ./outputs/training_data/train_simple_svg.jsonl
```

### 最终训练（adapter，r32 / alpha 64 / epochs 5 / cosine + warmup）

```powershell
D:/anaconda/envs/pytorch/python.exe -m student_kit.train_peft --model-path ./gemma3-270m --train-path ./outputs/training_data/train_simple_svg.jsonl --valid-path ./logo-detailed-prompt/valid.jsonl --epochs 5 --learning-rate 2e-4 --batch-size 1 --gradient-accumulation-steps 4 --max-length 1536 --lora-rank 32 --lora-alpha 64 --eval-steps 50 --save-steps 50 --output-dir ./runs/peft-gemma3-final-r32 --adapter-dir ./adapter --precision auto --gradient-checkpointing
```

### 最终评估（生成 + reward + reference 相似度）

```powershell
# adapter 生成 + reward 评分
D:/anaconda/envs/pytorch/python.exe -m student_kit.eval_self --mode generate --model-path ./gemma3-270m --adapter-path ./adapter --temperature 0 --top-p 1 --max-new-tokens 2048 --output ./outputs/results_adapter.json --generations-path ./outputs/generated_adapter.jsonl

# adapter reference 相似度对比
D:/anaconda/envs/pytorch/python.exe student_kit/compare_reference.py --predictions-jsonl ./outputs/generated_adapter.jsonl --output ./outputs/reference_similarity_adapter.json

# 基模型对比（用于第 5 节对比表）
D:/anaconda/envs/pytorch/python.exe -m student_kit.eval_self --mode generate --model-path ./gemma3-270m --temperature 0 --top-p 1 --max-new-tokens 2048 --output ./outputs/results_base.json --generations-path ./outputs/generated_base.jsonl
D:/anaconda/envs/pytorch/python.exe student_kit/compare_reference.py --predictions-jsonl ./outputs/generated_base.jsonl --output ./outputs/reference_similarity_base.json
```

## 文件清单

核心文件已就绪：

- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`
- `reward.py`
- `train_config.yaml`
- `results.json`
- `report.md`
