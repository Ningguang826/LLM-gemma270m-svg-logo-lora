"""比较生成 SVG 与 valid.jsonl 参考 SVG 的轻量相似度。

该指标不参与训练，只用于报告中补充说明生成结果是否更接近 reference：
颜色集合 Jaccard 反映配色接近度，标签集合 Jaccard 反映图元类型接近度。
扩展版额外统计：元素数量比值、结构化标签 Jaccard（含嵌套层级）、
gradient/filter 使用率、生成 SVG 颜色在 reference 主色集中的命中率。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET
from typing import Any


def main() -> None:
    args = parse_args()
    references = [get_message(row, "assistant") for row in read_jsonl(args.valid_path)]
    predictions = read_jsonl(args.predictions_jsonl)
    rows = []
    for index, prediction in enumerate(predictions):
        item_id = int(prediction.get("id", index))
        svg = str(prediction.get("svg", ""))
        reference = references[item_id]
        rows.append(compare_pair(item_id, svg, reference))
    output = {
        "metadata": {
            "valid_path": str(args.valid_path),
            "predictions_jsonl": str(args.predictions_jsonl),
        },
        "summary": summarize(rows),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"已写入: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-path", type=Path, default=Path("logo-detailed-prompt/valid.jsonl"))
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def get_message(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def compare_pair(item_id: int, svg: str, reference: str) -> dict[str, Any]:
    pred_colors = colors(svg)
    ref_colors = colors(reference)
    pred_tags = set(tags(svg))
    ref_tags = set(tags(reference))
    return {
        "id": item_id,
        "color_jaccard": jaccard(pred_colors, ref_colors),
        "tag_jaccard": jaccard(pred_tags, ref_tags),
        "valid_xml": is_valid_xml(svg),
        "element_count_pred": count_elements(svg),
        "element_count_ref": count_elements(reference),
        "element_count_ratio": safe_ratio(count_elements(svg), count_elements(reference)),
        "structural_tag_jaccard": jaccard(structural_tags(svg), structural_tags(reference)),
        "pred_uses_gradient": uses_gradient(svg),
        "ref_uses_gradient": uses_gradient(reference),
        "pred_uses_filter": uses_filter(svg),
        "ref_uses_filter": uses_filter(reference),
        "ref_color_hit_rate": ref_color_hit_rate(pred_colors, ref_colors),
    }


def colors(svg: str) -> set[str]:
    return {color.upper()[:7] for color in re.findall(r"#[0-9a-fA-F]{6}\b", svg or "")}


def tags(svg: str) -> list[str]:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return []
    return [element.tag.rsplit("}", 1)[-1] for element in root.iter()]


def structural_tags(svg: str) -> set[str]:
    """反映嵌套/分组/渐变/滤镜使用情况的结构标签集合。"""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return set()
    wanted = {"g", "defs", "linearGradient", "radialGradient", "filter", "stop",
              "feDropShadow", "feGaussianBlur", "feOffset", "feBlend", "symbol", "use"}
    return {element.tag.rsplit("}", 1)[-1] for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] in wanted}


def count_elements(svg: str) -> int:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return 0
    return sum(1 for _ in root.iter()) - 1  # 去掉 root 本身


def uses_gradient(svg: str) -> bool:
    return bool(re.search(r"<(?:linear|radial)Gradient\b", svg or "", flags=re.IGNORECASE))


def uses_filter(svg: str) -> bool:
    return bool(re.search(r"<filter\b", svg or "", flags=re.IGNORECASE))


def ref_color_hit_rate(pred_colors: set[str], ref_colors: set[str]) -> float:
    """生成 SVG 颜色集合中，有多少出现在 reference 颜色集合里。"""
    if not pred_colors:
        return 0.0
    return len(pred_colors & ref_colors) / len(pred_colors)


def is_valid_xml(svg: str) -> bool:
    try:
        ET.fromstring(svg)
    except ET.ParseError:
        return False
    return True


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def safe_ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return round(statistics.fmean(row[key] for row in rows), 6)

    def mean_int(key: str) -> float:
        return round(statistics.fmean(row[key] for row in rows), 3)

    return {
        "count": len(rows),
        "mean_color_jaccard": mean("color_jaccard"),
        "mean_tag_jaccard": mean("tag_jaccard"),
        "mean_structural_tag_jaccard": mean("structural_tag_jaccard"),
        "mean_element_count_pred": mean_int("element_count_pred"),
        "mean_element_count_ref": mean_int("element_count_ref"),
        "mean_element_count_ratio": mean("element_count_ratio"),
        "pred_gradient_count": sum(1 for row in rows if row["pred_uses_gradient"]),
        "ref_gradient_count": sum(1 for row in rows if row["ref_uses_gradient"]),
        "pred_filter_count": sum(1 for row in rows if row["pred_uses_filter"]),
        "ref_filter_count": sum(1 for row in rows if row["ref_uses_filter"]),
        "mean_ref_color_hit_rate": mean("ref_color_hit_rate"),
        "valid_xml_count": sum(1 for row in rows if row["valid_xml"]),
    }


if __name__ == "__main__":
    main()
