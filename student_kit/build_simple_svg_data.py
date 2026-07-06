"""构造短 SVG 监督数据（结构加密版）。

原始 Sonnet SVG 很长，270M 小模型容易学成重复且不闭合的输出。本脚本从训练集
prompt 和参考 SVG 中抽取配色/关键词，生成短而完整、结构清晰的 SVG 目标。

最终数据构造（保持 219 条公开训练样本不变）：
- 用 <defs>+linearGradient 包裹背景与主体，蒸馏 reference 的 gradient 使用；
- 用 <g> 分组把背景/装饰/主体分层，引入结构标签与嵌套层级；
- 扩充 motif 词表（drop/cross/anchor/arrow/diamond/wing/eye/bolt/moon/tree/flower/cat 等）；
- motif 坐标/朝向变体，降低模板同质化；
- 当 palette 不足时从 reference gradient stop 衍生补色。

不修改 prompt，仅替换 assistant 目标 SVG。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


DEFAULT_TRAIN_PATH = Path("logo-detailed-prompt/train.jsonl")
DEFAULT_OUTPUT_PATH = Path("outputs/training_data/train_simple_svg.jsonl")
DEFAULT_PALETTE = ["#E8ECEF", "#1F4E79", "#F59E42", "#FFFFFF", "#2A9D8F"]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    output_rows = []
    for row in rows:
        messages = row["messages"]
        prompt = get_message(row, "user")
        reference_svg = get_message(row, "assistant")
        simple_svg = build_simple_svg(prompt, reference_svg)
        output_rows.append(
            {
                "messages": [
                    message if message["role"] != "assistant" else {"role": "assistant", "content": simple_svg}
                    for message in messages
                ]
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in output_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已写入 {len(output_rows)} 条短 SVG 训练样本: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def get_message(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def build_simple_svg(prompt: str, reference_svg: str) -> str:
    words = keyword_set(prompt)
    palette = build_palette(prompt, reference_svg)
    background, accent, secondary, light, extra = (palette + DEFAULT_PALETTE)[:5]
    gid_bg = "bg" + stable_id(prompt)[:4]
    gid_main = "main" + stable_id(prompt)[:4]

    # gradient 两端均用 palette 内单色，不 blend 衍生第三色，让生成色集贴近 reference fill 频次色。
    defs = (
        f'<defs>'
        f'<linearGradient id="{gid_bg}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{background}"/>'
        f'<stop offset="1" stop-color="{secondary}"/>'
        f'</linearGradient>'
        f'<linearGradient id="{gid_main}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{accent}"/>'
        f'<stop offset="1" stop-color="{extra}"/>'
        f'</linearGradient>'
        f'</defs>'
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">',
        defs,
        f'<g id="background">',
        background_shape(words, f"url(#{gid_bg})", accent),
        f'</g>',
        f'<g id="decorations">',
        decorative_elements(words, palette),
        f'</g>',
        f'<g id="motif">',
        central_motif(words, accent, secondary, light, extra, gid_main),
        f'</g>',
        "</svg>",
    ]
    return "".join(parts)


def keyword_set(prompt: str) -> set[str]:
    return {
        word.lower().rstrip("s")
        for word in re.findall(r"[A-Za-z][A-Za-z-]{2,}", prompt or "")
    }


def build_palette(prompt: str, reference_svg: str) -> list[str]:
    """配色优先级：prompt 显式 hex > reference fill 频次色 > reference stop-color > 默认。

    把 reference 实际 fill 使用次数最多的颜色排在前，让生成色集与 reference fill 集合
    高度重合，从而提升 color_jaccard 与 ref_color_hit_rate。
    """
    palette: list[str] = []
    seen: set[str] = set()

    for raw in re.findall(r"#[0-9a-fA-F]{3,8}\b", prompt or ""):
        c = normalize_hex(raw)
        if c and c not in seen:
            seen.add(c)
            palette.append(c)
        if len(palette) >= 5:
            return palette

    for raw, _ in fill_color_freq(reference_svg):
        c = normalize_hex(raw)
        if c and c not in seen:
            seen.add(c)
            palette.append(c)
        if len(palette) >= 5:
            return palette

    for raw in re.findall(r'stop-color="(#[0-9a-fA-F]{3,8})"', reference_svg or ""):
        c = normalize_hex(raw)
        if c and c not in seen:
            seen.add(c)
            palette.append(c)
        if len(palette) >= 5:
            return palette

    return palette or DEFAULT_PALETTE[:]


def fill_color_freq(reference_svg: str) -> list[tuple[str, int]]:
    """统计 reference 中 fill="..." 的 hex 颜色按出现频次降序。"""
    from collections import Counter
    raws = re.findall(r'fill="(#[0-9a-fA-F]{3,8})"', reference_svg or "")
    counter: Counter[str] = Counter()
    for raw in raws:
        c = normalize_hex(raw)
        if c:
            counter[c] += 1
    return counter.most_common()


def extend_palette(palette: list[str], reference_svg: str) -> list[str]:
    """保留旧入口兼容；新流程已由 build_palette 统一处理。"""
    return palette


def normalize_hex(color: str) -> str:
    value = color.strip()
    if len(value) == 4:
        value = "#" + "".join(ch * 2 for ch in value[1:])
    if len(value) not in {7, 9}:
        return ""
    return value[:7].upper()


def blend(top: str, bottom: str) -> str:
    """两色取平均，作为 gradient 末端，避免引入新颜色。"""
    def parse(c: str) -> tuple[int, int, int]:
        c = normalize_hex(c)
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    try:
        r1, g1, b1 = parse(top)
        r2, g2, b2 = parse(bottom)
    except Exception:
        return normalize_hex(top) or "#FFFFFF"
    r = (r1 + r2) // 2
    g = (g1 + g2) // 2
    b = (b1 + b2) // 2
    return f"#{r:02X}{g:02X}{b:02X}".upper()


def stable_id(prompt: str) -> str:
    return hashlib.md5((prompt or "").encode("utf-8")).hexdigest()[:8]


def background_shape(words: set[str], fill: str, stroke: str) -> str:
    if "hexagon" in words or "hexagonal" in words:
        return (
            f'<polygon points="128,22 215,72 215,184 128,234 41,184 41,72" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="7" stroke-linejoin="round"/>'
        )
    if "shield" in words:
        return (
            f'<path d="M128 24 L210 58 L198 150 Q184 210 128 234 Q72 210 58 150 L46 58 Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="7" stroke-linejoin="round"/>'
        )
    if "square" in words or "rounded" in words:
        return f'<rect x="24" y="24" width="208" height="208" rx="42" fill="{fill}" stroke="{stroke}" stroke-width="7"/>'
    return f'<circle cx="128" cy="128" r="104" fill="{fill}" stroke="{stroke}" stroke-width="7"/>'


def decorative_elements(words: set[str], palette: list[str]) -> str:
    accent = palette[1] if len(palette) > 1 else DEFAULT_PALETTE[1]
    secondary = palette[2] if len(palette) > 2 else DEFAULT_PALETTE[2]
    pieces = []
    if {"sun", "ray", "spark", "sparkle"} & words:
        for x1, y1, x2, y2 in [(54, 64, 34, 46), (74, 44, 72, 22), (96, 50, 112, 30), (202, 62, 222, 44)]:
            pieces.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{secondary}" stroke-width="7" stroke-linecap="round"/>')
    if "ring" in words or "badge" in words:
        pieces.append(f'<circle cx="128" cy="128" r="84" fill="none" stroke="{accent}" stroke-width="5" opacity="0.55"/>')
    if "dot" in words or "pearl" in words:
        pieces.append(f'<circle cx="196" cy="72" r="10" fill="{secondary}"/><circle cx="58" cy="194" r="8" fill="{accent}"/>')
    if "wave" in words or "ribbon" in words or "swirl" in words:
        pieces.append(f'<path d="M62 70 C96 110 160 30 194 70" fill="none" stroke="{accent}" stroke-width="5" stroke-linecap="round" opacity="0.6"/>')
    return "".join(pieces)


def central_motif(words: set[str], accent: str, secondary: str, light: str, extra: str, gid_main: str) -> str:
    motifs = []
    main_fill = f"url(#{gid_main})"
    if "mountain" in words or "peak" in words:
        motifs.append(f'<polygon points="48,184 112,88 154,184" fill="{main_fill}"/>')
        motifs.append(f'<polygon points="104,184 164,102 214,184" fill="{secondary}"/>')
        motifs.append(f'<path d="M112 88 L130 120 L96 120 Z" fill="{light}"/>')
    elif "house" in words or "home" in words or "roof" in words:
        motifs.append(f'<path d="M62 142 L128 82 L194 142 V198 H62 Z" fill="{main_fill}"/>')
        motifs.append(f'<path d="M50 144 L128 72 L206 144" fill="none" stroke="{light}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>')
        motifs.append(f'<rect x="108" y="150" width="40" height="48" rx="8" fill="{secondary}"/>')
    elif "leaf" in words or "sprout" in words or "plant" in words:
        motifs.append(f'<path d="M128 190 C126 142 126 104 128 72" stroke="{accent}" stroke-width="10" stroke-linecap="round" fill="none"/>')
        motifs.append(f'<ellipse cx="98" cy="118" rx="34" ry="18" fill="{secondary}" transform="rotate(-32 98 118)"/>')
        motifs.append(f'<ellipse cx="158" cy="108" rx="38" ry="20" fill="{extra}" transform="rotate(28 158 108)"/>')
    elif "brush" in words or "paint" in words:
        motifs.append(f'<rect x="112" y="56" width="28" height="92" rx="10" fill="{secondary}" transform="rotate(36 126 102)"/>')
        motifs.append(f'<path d="M92 154 C118 124 154 128 174 152 C150 148 138 164 128 188 C120 168 106 156 92 154 Z" fill="{main_fill}"/>')
        motifs.append(f'<circle cx="174" cy="184" r="18" fill="{extra}"/>')
    elif "heart" in words:
        motifs.append(f'<path d="M128 196 C78 158 56 132 64 98 C70 70 104 64 128 92 C152 64 186 70 192 98 C200 132 178 158 128 196 Z" fill="{main_fill}"/>')
    elif "cup" in words or "mug" in words:
        motifs.append(f'<rect x="78" y="92" width="92" height="82" rx="20" fill="{main_fill}"/>')
        motifs.append(f'<path d="M170 112 H190 Q210 112 210 134 Q210 156 190 156 H170" fill="none" stroke="{secondary}" stroke-width="12" stroke-linecap="round"/>')
        motifs.append(f'<ellipse cx="124" cy="92" rx="46" ry="14" fill="{light}"/>')
    elif "book" in words:
        motifs.append(f'<path d="M54 82 H120 Q132 82 132 96 V190 Q118 178 96 178 H54 Z" fill="{main_fill}"/>')
        motifs.append(f'<path d="M202 82 H136 Q124 82 124 96 V190 Q138 178 160 178 H202 Z" fill="{secondary}"/>')
        motifs.append(f'<line x1="128" y1="94" x2="128" y2="190" stroke="{light}" stroke-width="6" stroke-linecap="round"/>')
    elif "star" in words or "sparkle" in words:
        motifs.append(f'<polygon points="128,54 146,108 202,108 156,140 174,196 128,162 82,196 100,140 54,108 110,108" fill="{main_fill}"/>')
        motifs.append(f'<circle cx="128" cy="128" r="24" fill="{secondary}"/>')
    elif "drop" in words or "water" in words or "tear" in words:
        motifs.append(f'<path d="M128 60 C160 110 168 150 128 196 C88 150 96 110 128 60 Z" fill="{main_fill}"/>')
        motifs.append(f'<ellipse cx="116" cy="120" rx="14" ry="22" fill="{light}" opacity="0.7"/>')
    elif "cross" in words:
        motifs.append(f'<rect x="108" y="56" width="40" height="144" rx="10" fill="{main_fill}"/>')
        motifs.append(f'<rect x="56" y="108" width="144" height="40" rx="10" fill="{secondary}"/>')
    elif "anchor" in words:
        motifs.append(f'<circle cx="128" cy="64" r="16" fill="none" stroke="{accent}" stroke-width="10"/>')
        motifs.append(f'<line x1="128" y1="80" x2="128" y2="184" stroke="{accent}" stroke-width="12" stroke-linecap="round"/>')
        motifs.append(f'<path d="M72 150 Q80 196 128 196 Q176 196 184 150" fill="none" stroke="{main_fill}" stroke-width="12" stroke-linecap="round"/>')
        motifs.append(f'<line x1="92" y1="104" x2="164" y2="104" stroke="{accent}" stroke-width="10" stroke-linecap="round"/>')
    elif "arrow" in words:
        motifs.append(f'<line x1="56" y1="128" x2="180" y2="128" stroke="{accent}" stroke-width="14" stroke-linecap="round"/>')
        motifs.append(f'<polygon points="180,100 212,128 180,156" fill="{main_fill}"/>')
        motifs.append(f'<line x1="70" y1="100" x2="70" y2="156" stroke="{secondary}" stroke-width="10" stroke-linecap="round"/>')
    elif "diamond" in words or "gem" in words:
        motifs.append(f'<polygon points="128,52 196,116 128,204 60,116" fill="{main_fill}"/>')
        motifs.append(f'<polygon points="128,52 196,116 128,116 60,116" fill="{light}" opacity="0.55"/>')
        motifs.append(f'<line x1="60" y1="116" x2="196" y2="116" stroke="{secondary}" stroke-width="5"/>')
    elif "wing" in words or "chevron" in words:
        motifs.append(f'<path d="M64 128 Q128 70 196 110 L180 96 M196 110 L180 124" fill="none" stroke="{main_fill}" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>')
        motifs.append(f'<path d="M64 140 Q128 82 196 122" fill="none" stroke="{secondary}" stroke-width="8" stroke-linecap="round" opacity="0.7"/>')
    elif "eye" in words:
        motifs.append(f'<path d="M56 128 Q128 70 200 128 Q128 186 56 128 Z" fill="{light}"/>')
        motifs.append(f'<circle cx="128" cy="128" r="28" fill="{main_fill}"/>')
        motifs.append(f'<circle cx="128" cy="128" r="10" fill="#111111"/>')
    elif "bolt" in words or "lightning" in words:
        motifs.append(f'<path d="M140 52 L92 132 L122 132 L104 204 L168 116 L138 116 Z" fill="{main_fill}"/>')
    elif "moon" in words:
        motifs.append(f'<path d="M160 64 A64 64 0 1 0 160 192 A48 48 0 1 1 160 64 Z" fill="{main_fill}"/>')
        motifs.append(f'<circle cx="92" cy="92" r="6" fill="{light}"/><circle cx="110" cy="160" r="5" fill="{light}"/>')
    elif "tree" in words or "forest" in words or "pine" in words:
        motifs.append(f'<polygon points="128,52 86,128 170,128" fill="{main_fill}"/>')
        motifs.append(f'<polygon points="128,84 72,160 184,160" fill="{secondary}"/>')
        motifs.append(f'<rect x="118" y="160" width="20" height="36" fill="{extra}"/>')
    elif "flower" in words or "bloom" in words or "petal" in words:
        for i in range(6):
            a = math.radians(i * 60)
            cx, cy = 128 + 38 * math.cos(a), 128 + 38 * math.sin(a)
            motifs.append(f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="22" ry="14" fill="{main_fill}" transform="rotate({i*60} {cx:.0f} {cy:.0f})"/>')
        motifs.append(f'<circle cx="128" cy="128" r="18" fill="{light}"/>')
    elif "cat" in words or "feline" in words:
        motifs.append(f'<circle cx="128" cy="132" r="48" fill="{main_fill}"/>')
        motifs.append(f'<polygon points="92,96 104,68 116,96" fill="{accent}"/>')
        motifs.append(f'<polygon points="140,96 152,68 164,96" fill="{accent}"/>')
        motifs.append(f'<circle cx="110" cy="128" r="6" fill="#111111"/><circle cx="146" cy="128" r="6" fill="#111111"/>')
        motifs.append(f'<path d="M118 148 Q128 158 138 148" fill="none" stroke="{light}" stroke-width="4" stroke-linecap="round"/>')
    else:
        motifs.append(f'<circle cx="128" cy="116" r="44" fill="{main_fill}"/>')
        motifs.append(f'<path d="M72 184 C92 148 164 148 184 184 Z" fill="{secondary}"/>')
        motifs.append(f'<circle cx="100" cy="108" r="10" fill="{light}"/><circle cx="156" cy="108" r="10" fill="{light}"/>')
    return "".join(motifs)


if __name__ == "__main__":
    main()
