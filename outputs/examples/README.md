# 示例文件说明

本目录保留七条代表性 valid 样本（id=1/3/5/6/8/12/13）的对比，覆盖配色命中、部分命中、回退默认三种典型：

- `*_prompt.txt`：来自 `logo-detailed-prompt/valid.jsonl` 中对应样本的 `user` 字段。
- `*_reference.svg`：同样本的 `assistant` 字段，数据集参考 SVG。
- `*_adapter.svg`：最终 adapter（r32 + 短 SVG 蒸馏数据）在 temperature=0 下生成的 SVG。

| id | prompt 含 hex | color_jaccard | 配色定性 |
|---:|:---:|---:|---|
| 1 | ✓ | 0.75 | 命中 |
| 3 | ✓ | 0.44 | 部分命中 |
| 5 | ✓ | 0.60 | 命中 |
| 12 | ✓ | 0.80 | 高保真 |
| 6 | ✗ | 0.08 | 回退默认 |
| 8 | ✗ | 0.00 | 回退默认 |
| 13 | ✗ | 0.00 | 回退默认 |

最终 adapter 在结构密度与图元类型上表现稳定（七条均含 `<defs>/<g>/linearGradient`）；配色保真度严格随"prompt 是否含显式 hex"二分——含 hex 时 jaccard 0.44–0.80，不含 hex 时回退默认暖色 0.00–0.08，根因是 270M×219 条数据的语义→hex 泛化天花板。
