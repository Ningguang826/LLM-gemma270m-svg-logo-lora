"""SVG 徽标生成任务的可解释奖励函数。

这个 reward 是训练代理指标，不试图替代人工审美；它优先奖励安全、有效、
不过度退化、坐标合理的 SVG，再用轻量关键词覆盖度衡量提示词响应。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import statistics
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
MAX_OUTPUT_CHARS = 12000
CANVAS_MIN = -8.0
CANVAS_MAX = 264.0
IDEAL_MIN = 16.0
IDEAL_MAX = 240.0

ALLOWED_TAGS = {
    "svg", "defs", "g", "path", "circle", "ellipse", "rect", "polygon",
    "polyline", "line", "linearGradient", "radialGradient", "stop", "filter",
    "feDropShadow", "feGaussianBlur", "feOffset", "feBlend",
}
BLOCKED_TAGS = {"script", "foreignObject", "image", "iframe", "object", "embed", "style", "a"}
SHAPE_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}
COLOR_ATTRS = {"fill", "stroke", "stop-color"}
NUMERIC_ATTRS = {
    "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
    "width", "height", "stroke-width",
}

GENERIC_PROMPT_WORDS = {
    "and", "the", "with", "from", "into", "that", "this", "logo", "shape",
    "center", "centered", "clean", "style", "filled", "behind", "inside",
    "everything", "composition", "small", "main", "like", "giving",
}
KEYWORD_TO_SVG_HINTS = {
    "circle": {"circle", "ellipse", "path"},
    "circular": {"circle", "ellipse", "path"},
    "badge": {"circle", "rect", "path"},
    "square": {"rect", "path"},
    "rounded": {"rect", "path"},
    "ring": {"circle", "ellipse", "path"},
    "dot": {"circle", "ellipse"},
    "line": {"line", "path", "polyline"},
    "ray": {"line", "path"},
    "leaf": {"path", "ellipse"},
    "sprout": {"path", "ellipse"},
    "ribbon": {"path"},
    "swirl": {"path"},
    "star": {"polygon", "path"},
    "sparkle": {"polygon", "path"},
    "brush": {"rect", "path"},
    "mountain": {"polygon", "path"},
    "shield": {"path", "polygon"},
    "heart": {"path"},
    "hexagon": {"polygon", "path"},
    "hexagonal": {"polygon", "path"},
    "triangle": {"polygon", "path"},
    "house": {"path", "polygon", "rect"},
    "home": {"path", "polygon", "rect"},
    "roof": {"path", "polygon"},
    "book": {"path", "rect"},
    "cup": {"rect", "ellipse", "path"},
    "mug": {"rect", "ellipse", "path"},
    "cloud": {"circle", "ellipse", "path"},
    "flame": {"path"},
    "wave": {"path"},
    "sun": {"circle", "line", "path"},
}
COLOR_WORDS = {
    "black", "blue", "brown", "coral", "cream", "gold", "golden", "green",
    "grey", "gray", "navy", "orange", "pink", "purple", "red", "teal",
    "white", "yellow",
}


@dataclass(frozen=True)
class RewardBreakdown:
    score: float
    subscores: dict[str, float]
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 6),
            "subscores": {key: round(value, 6) for key, value in self.subscores.items()},
            "reasons": self.reasons,
        }


def score_svg(svg_text: str, prompt: str = "") -> RewardBreakdown:
    """返回总分、子分数和原因列表，方便训练后写分析报告。"""

    reasons: list[str] = []
    text = (svg_text or "").strip()
    if not text:
        return _zero("empty_output")

    extracted = extract_svg(text)
    if extracted != text:
        reasons.append("non_svg_wrapping_text")
    text = extracted
    length_score = _score_length(text, reasons)

    if not text.lower().startswith("<svg") or "</svg>" not in text.lower():
        reasons.append("missing_svg_envelope")
        return _weighted(_base_subscores(length=length_score), reasons)

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        reasons.append("xml_parse_error")
        subscores = _base_subscores(length=length_score)
        subscores["safe_svg"] = 0.15
        return _weighted(subscores, reasons)

    elements = list(root.iter())
    tags = [_local_name(element.tag) for element in elements]
    shapes = [element for element in elements if _local_name(element.tag) in SHAPE_TAGS]
    colors = _collect_colors(elements)
    numbers = _collect_numeric_values(elements)

    subscores = {
        "valid_xml": _score_valid_xml(root, reasons),
        "safe_svg": _score_safety(elements, reasons),
        "viewbox": _score_viewbox(root, reasons),
        "structure": _score_structure(tags, shapes, reasons),
        "geometry": _score_geometry(numbers, reasons),
        "palette": _score_palette(colors, shapes, reasons, prompt=prompt),
        "non_degenerate": _score_non_degenerate(text, tags, shapes, reasons),
        "prompt_alignment": _score_prompt_alignment(prompt, tags, colors, reasons),
        "length": length_score,
    }
    return _weighted(subscores, reasons)

def reward(svg_text: str, prompt: str = "") -> float:
    return score_svg(svg_text, prompt).score


def score(svg_text: str, prompt: str = "") -> float:
    return reward(svg_text, prompt)


def score_batch(items: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        score_svg(item.get("svg", ""), item.get("prompt", "")).to_dict()
        for item in items
    ]


def extract_svg(text: str) -> str:
    match = re.search(r"<svg\b[\s\S]*?</svg>", text or "", flags=re.IGNORECASE)
    return match.group(0).strip() if match else (text or "").strip()


def aggregate_scores(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"count": 0, "mean_score": 0.0, "subscores": {}}
    scores = [float(row["score"]) for row in rows]
    keys = sorted({key for row in rows for key in row.get("subscores", {})})
    subscores = {}
    for key in keys:
        values = [float(row["subscores"].get(key, 0.0)) for row in rows]
        subscores[key] = {
            "mean": round(statistics.fmean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    return {
        "count": len(rows),
        "mean_score": round(statistics.fmean(scores), 6),
        "min_score": round(min(scores), 6),
        "max_score": round(max(scores), 6),
        "subscores": subscores,
    }


def _zero(reason: str) -> RewardBreakdown:
    return RewardBreakdown(0.0, _base_subscores(length=0.0), [reason])


def _base_subscores(length: float) -> dict[str, float]:
    return {
        "valid_xml": 0.0,
        "safe_svg": 0.0,
        "viewbox": 0.0,
        "structure": 0.0,
        "geometry": 0.0,
        "palette": 0.0,
        "non_degenerate": 0.0,
        "prompt_alignment": 0.0,
        "length": length,
    }


def _weighted(subscores: dict[str, float], reasons: list[str]) -> RewardBreakdown:
    weights = {
        "valid_xml": 0.20,
        "safe_svg": 0.13,
        "viewbox": 0.10,
        "structure": 0.14,
        "geometry": 0.12,
        "palette": 0.09,
        "non_degenerate": 0.12,
        "prompt_alignment": 0.12,
        "length": 0.02,
    }
    total = sum(subscores[key] * weight for key, weight in weights.items())
    return RewardBreakdown(max(0.0, min(1.0, total)), subscores, reasons)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _score_length(text: str, reasons: list[str]) -> float:
    length = len(text)
    if length > MAX_OUTPUT_CHARS:
        reasons.append("too_long")
        return 0.0
    if length < 80:
        reasons.append("too_short")
        return 0.2
    if 250 <= length <= 6000:
        return 1.0
    return 0.65 if length < 250 else 0.75


def _score_valid_xml(root: ET.Element, reasons: list[str]) -> float:
    if _local_name(root.tag) != "svg":
        reasons.append("root_not_svg")
        return 0.0
    return 1.0


def _score_safety(elements: list[ET.Element], reasons: list[str]) -> float:
    value = 1.0
    for element in elements:
        tag = _local_name(element.tag)
        if tag in BLOCKED_TAGS:
            reasons.append(f"blocked_tag:{tag}")
            value -= 0.5
        elif tag not in ALLOWED_TAGS:
            reasons.append(f"unknown_tag:{tag}")
            value -= 0.12
        for attr, raw in element.attrib.items():
            name = _local_name(attr).lower()
            text = str(raw).lower()
            if name.startswith("on"):
                reasons.append(f"event_handler_attr:{name}")
                value -= 0.5
            if any(marker in text for marker in ("javascript:", "data:", "http://", "https://")):
                reasons.append(f"external_or_executable_ref:{name}")
                value -= 0.35
    return max(0.0, min(1.0, value))


def _score_viewbox(root: ET.Element, reasons: list[str]) -> float:
    viewbox = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    value = 0.0
    if viewbox:
        nums = _numbers(viewbox)
        if len(nums) == 4 and all(abs(a - b) <= 0.01 for a, b in zip(nums, [0, 0, 256, 256])):
            value += 0.8
        elif len(nums) == 4 and nums[2] > 0 and nums[3] > 0:
            value += 0.45
            reasons.append("non_standard_viewbox")
        else:
            reasons.append("invalid_viewbox")
    else:
        reasons.append("missing_viewbox")
    xmlns = root.attrib.get("xmlns")
    has_svg_namespace = root.tag.startswith(f"{{{SVG_NS}}}")
    if xmlns == SVG_NS or has_svg_namespace:
        value += 0.2
    elif xmlns:
        value += 0.1
        reasons.append("non_standard_xmlns")
    else:
        reasons.append("missing_xmlns")
    return min(1.0, value)


def _score_structure(tags: list[str], shapes: list[ET.Element], reasons: list[str]) -> float:
    count = len(shapes)
    value = 0.0
    if count == 0:
        reasons.append("no_shape_elements")
    elif 3 <= count <= 40:
        value += 0.6
    elif count < 3:
        value += 0.3
        reasons.append("too_few_shapes")
    else:
        value += 0.4
        reasons.append("too_many_shapes")
    value += 0.2 if tags.count("svg") == 1 else 0.0
    if tags.count("svg") != 1:
        reasons.append("multiple_or_missing_svg_tags")
    if {"path", "circle", "rect"} & set(tags):
        value += 0.2
    return min(1.0, value)


def _score_geometry(values: list[float], reasons: list[str]) -> float:
    if not values:
        reasons.append("no_numeric_geometry")
        return 0.25
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        reasons.append("non_finite_number")
        return 0.0
    soft = sum(CANVAS_MIN <= value <= CANVAS_MAX for value in finite) / len(finite)
    ideal = sum(IDEAL_MIN <= value <= IDEAL_MAX for value in finite) / len(finite)
    value = 0.75 * soft + 0.25 * min(1.0, ideal * 1.35)
    if soft < 0.85:
        reasons.append("many_coordinates_out_of_canvas")
    if max(abs(value) for value in finite) > 2000:
        reasons.append("extreme_coordinate")
        value *= 0.35
    return max(0.0, min(1.0, value))


def _score_palette(colors: list[str], shapes: list[ET.Element], reasons: list[str], prompt: str = "") -> float:
    if not shapes:
        return 0.0
    normalized = [_normalize_color(color) for color in colors if _normalize_color(color)]
    unique = set(normalized)
    if not normalized:
        reasons.append("missing_colors")
        return 0.35
    if 2 <= len(unique) <= 7:
        base = 1.0
    elif len(unique) == 1:
        reasons.append("single_color_palette")
        base = 0.68
    else:
        reasons.append("too_many_colors")
        base = 0.72
    # 若 prompt 显式给出了 hex 颜色但输出未复用，扣分——鼓励贴近提示词配色。
    prompt_colors = {_normalize_color(c) for c in re.findall(r"#[0-9a-fA-F]{3,8}\b", prompt or "")}
    prompt_colors.discard("")
    if prompt_colors:
        hit = len(prompt_colors & unique) / len(prompt_colors)
        if hit < 0.34:
            reasons.append("missed_prompt_palette")
            base *= 0.85 + 0.15 * hit
    return max(0.0, min(1.0, base))


def _score_non_degenerate(text: str, tags: list[str], shapes: list[ET.Element], reasons: list[str]) -> float:
    if not shapes:
        return 0.0
    signatures = {(_local_name(element.tag), tuple(sorted(element.attrib.items()))) for element in shapes}
    unique_ratio = len(signatures) / len(shapes)
    repeated_pressure = max(Counter(tags).values()) / max(1, len(tags))
    value = 0.45 + 0.35 * min(1.0, unique_ratio * 1.2)
    value += 0.1 if repeated_pressure < 0.85 else 0.0
    value += 0.1 if len(set(text.split())) > 6 else 0.0
    if unique_ratio < 0.35:
        reasons.append("repetitive_shapes")
    if re.search(r"(.)\1{80,}", text):
        reasons.append("repeated_character_run")
        value *= 0.2
    return max(0.0, min(1.0, value))


def _score_prompt_alignment(prompt: str, tags: list[str], colors: list[str], reasons: list[str]) -> float:
    if not prompt:
        return 0.5
    words = _keywords(prompt)
    tag_set = set(tags)
    normalized_colors = {_normalize_color(color) for color in colors}
    color_families = {_color_family(color) for color in normalized_colors}
    matched = 0
    expected = 0
    for word in words:
        hints = KEYWORD_TO_SVG_HINTS.get(word)
        if hints:
            expected += 1
            matched += 1 if tag_set & hints else 0
        elif word in COLOR_WORDS:
            expected += 1
            # 训练数据和模型输出大量使用十六进制颜色，因此不能只靠字符串包含判断。
            aliases = _color_word_aliases(word)
            matched += 1 if any(word in color for color in normalized_colors) or aliases & color_families else 0
    if expected == 0:
        return 0.55
    value = matched / expected
    if value < 0.3:
        reasons.append("low_prompt_keyword_coverage")
    return max(0.0, min(1.0, 0.25 + 0.75 * value))


def _collect_colors(elements: list[ET.Element]) -> list[str]:
    colors: list[str] = []
    for element in elements:
        for attr, value in element.attrib.items():
            name = _local_name(attr)
            if name in COLOR_ATTRS:
                colors.append(str(value))
            if name == "style":
                colors.extend(re.findall(r"(?:fill|stroke|stop-color)\s*:\s*([^;]+)", str(value)))
    return colors


def _collect_numeric_values(elements: list[ET.Element]) -> list[float]:
    values: list[float] = []
    for element in elements:
        if _local_name(element.tag) not in SHAPE_TAGS:
            continue
        for attr, raw in element.attrib.items():
            name = _local_name(attr)
            if name in NUMERIC_ATTRS or name in {"points", "d", "transform"}:
                values.extend(_numbers(str(raw)))
    return values


def _numbers(text: str) -> list[float]:
    result = []
    for match in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text or ""):
        try:
            result.append(float(match))
        except ValueError:
            continue
    return result


def _normalize_color(color: str) -> str:
    value = (color or "").strip().lower()
    if not value or value in {"none", "transparent", "currentcolor"}:
        return ""
    if value.startswith("url("):
        return "gradient"
    return value


def _color_family(color: str) -> str:
    value = (color or "").strip().lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", value):
        return ""
    red = int(value[1:3], 16)
    green = int(value[3:5], 16)
    blue = int(value[5:7], 16)
    max_channel = max(red, green, blue)
    min_channel = min(red, green, blue)
    if max_channel < 45:
        return "black"
    if min_channel > 215:
        return "white"
    if max_channel - min_channel < 25:
        return "gray"
    if red > 190 and green > 150 and blue < 120:
        return "yellow" if green > 180 else "orange"
    if red > 180 and green < 120 and blue < 140:
        return "red"
    if red > 170 and blue > 150 and green < 150:
        return "pink" if red > blue else "purple"
    if green > red and green > blue:
        return "teal" if blue > 110 else "green"
    if blue > red and blue > green:
        return "navy" if blue < 130 else "blue"
    if red > green > blue:
        return "brown" if red < 170 else "gold"
    return ""


def _color_word_aliases(word: str) -> set[str]:
    aliases = {
        "grey": {"gray"},
        "gray": {"gray"},
        "cream": {"white", "yellow"},
        "coral": {"red", "orange", "pink"},
        "golden": {"gold", "yellow", "orange"},
        "gold": {"gold", "yellow", "orange"},
        "navy": {"navy", "blue"},
    }
    return aliases.get(word, {word})


def _keywords(prompt: str) -> set[str]:
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z-]{2,}", prompt or "")
        if word.lower() not in GENERIC_PROMPT_WORDS
    }
    return {word.rstrip("s") if len(word) > 4 else word for word in words}
