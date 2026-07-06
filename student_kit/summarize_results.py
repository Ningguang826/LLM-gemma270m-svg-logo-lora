"""汇总 base 与 adapter 的 results.json，生成报告可直接引用的数字。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    base_score = base["summary"]["mean_score"]
    adapter_score = adapter["summary"]["mean_score"]
    delta = round(adapter_score - base_score, 6)
    print(f"base_mean={base_score}")
    print(f"adapter_mean={adapter_score}")
    print(f"delta={delta}")


if __name__ == "__main__":
    main()

