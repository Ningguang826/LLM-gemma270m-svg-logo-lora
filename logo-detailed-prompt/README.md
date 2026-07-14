# logo-detailed-prompt SVG 数据集

本目录保存用于 SVG 徽标生成的公开 chat JSONL 数据。每条记录由详细视觉提示词和完整 SVG 响应组成，可直接作为监督微调数据使用。

## 文件

| 文件 | 行数 | 内容 |
|---|---:|---|
| `train.jsonl` | 219 | 训练样本 |
| `valid.jsonl` | 17 | 验证样本 |

每行均为一个 chat 格式样本：

```json
{"messages": [
  {"role": "system", "content": "<SVG-designer instructions>"},
  {"role": "user", "content": "<detailed visual prompt>"},
  {"role": "assistant", "content": "<complete <svg>...</svg>>"}
]}
```

- 输入为 `user` 字段中的详细视觉提示词。
- 目标为 `assistant` 字段中的完整 SVG 文档，使用 `viewBox="0 0 256 256"`。
- 训练时仅对 assistant 侧的 SVG token 计算损失。

## 来源

数据来源：`https://github.com/roboticcam/logo-detailed-prompt`。本仓库保留其公开训练集和验证集，并在根目录提供适配的训练、评估与奖励函数实现。