"""自评脚本：生成或读取 SVG 结果，并用本项目 reward 评分。

常用方式：
1. 只验证 reward：读取 valid.jsonl 中的参考 SVG。
   python student_kit/eval_self.py --mode reference

2. 评估已有预测文件，每行包含 {"id": ..., "svg": "..."}。
   python student_kit/eval_self.py --predictions-jsonl predictions.jsonl

3. 在 GPU 环境中生成并评估模型输出。
   python student_kit/eval_self.py --model-path ./gemma3-270m
   python student_kit/eval_self.py --model-path ./gemma3-270m --adapter-path ./adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
from typing import Any

from reward import aggregate_scores, extract_svg, score_svg
from student_kit.prompt_utils import build_prompt as build_prompt_shared


DEFAULT_VALID_PATH = Path("logo-detailed-prompt/valid.jsonl")
DEFAULT_OUTPUT_PATH = Path("results.json")
DEFAULT_GENERATIONS_PATH = Path("outputs/generated_valid.jsonl")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    multi_temp_settings = parse_multi_temp(args.multi_temp)
    if multi_temp_settings:
        run_multi_temp_eval(args, multi_temp_settings)
        return

    examples = read_chat_jsonl(args.valid_path)
    if args.limit:
        examples = examples[: args.limit]
    if args.mode == "reference":
        predictions = [
            {
                "id": index,
                "prompt": get_message(example, "user"),
                "svg": get_message(example, "assistant"),
                "source": "reference",
            }
            for index, example in enumerate(examples)
        ]
    elif args.predictions_jsonl:
        predictions = read_predictions(args.predictions_jsonl, examples)
    else:
        predictions = generate_predictions(args, examples)

    scored_rows = []
    for item in predictions:
        breakdown = score_svg(str(item.get("svg", "")), str(item.get("prompt", ""))).to_dict()
        scored_rows.append(
            {
                "id": item.get("id"),
                "source": item.get("source", "prediction"),
                "prompt": item.get("prompt", ""),
                "svg": item.get("svg", ""),
                **breakdown,
            }
        )

    output = {
        "metadata": {
            "valid_path": str(args.valid_path),
            "mode": args.mode,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": aggregate_scores(scored_rows),
        "samples": scored_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"已写入: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--mode", choices=["reference", "generate"], default="generate")
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--model-path", type=str, default="./gemma3-270m")
    parser.add_argument("--adapter-path", type=str)
    parser.add_argument("--generations-path", type=Path, default=DEFAULT_GENERATIONS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--multi-temp", type=str, default="",
                        help="逗号分隔的温度列表，如 '0.3,0.7'；启用多温度采样评估")
    parser.add_argument("--multi-temp-samples", type=int, default=4,
                        help="每个温度采样次数")
    parser.add_argument("--weak-samples-dir", type=Path, default=None,
                        help="若设置，将 prompt_alignment/color_jaccard 最低的样本 SVG 复制到此目录")
    return parser.parse_args()


def parse_multi_temp(raw: str) -> list[float]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        return [float(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        return []


def run_multi_temp_eval(args: argparse.Namespace, temps: list[float]) -> None:
    """对每个温度采样若干次，报告 reward 均值与方差。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = read_chat_jsonl(args.valid_path)
    if args.limit:
        examples = examples[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_dtype = resolve_generation_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=load_dtype, device_map="auto", trust_remote_code=True,
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rng = random.Random(args.seed)
    summary_rows = []
    all_samples: list[dict[str, Any]] = []
    for temp in temps:
        scores: list[float] = []
        for index, example in enumerate(examples):
            messages = [m for m in example["messages"] if m["role"] != "assistant"]
            prompt_text = build_prompt_shared(tokenizer, messages)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            for _ in range(args.multi_temp_samples):
                seed = int(rng.random() * 1_000_000)
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=temp > 0,
                        temperature=max(temp, 1e-5),
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
                svg = extract_svg(gen)
                bd = score_svg(svg, get_message(example, "user"))
                scores.append(bd.score)
                all_samples.append({
                    "id": index, "temperature": temp, "seed": seed,
                    "svg": svg, "score": bd.score,
                    "prompt": get_message(example, "user"),
                })
        summary_rows.append({
            "temperature": temp,
            "n": len(scores),
            "mean": round(statistics.fmean(scores), 6) if scores else 0.0,
            "min": round(min(scores), 6) if scores else 0.0,
            "max": round(max(scores), 6) if scores else 0.0,
            "std": round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0,
        })
    output = {"metadata": {"adapter_path": args.adapter_path, "temps": temps},
              "summary": summary_rows, "samples": all_samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"已写入多温度评估: {args.output}")

    if weak_path := args.weak_samples_dir:
        export_weak_samples(all_samples, weak_path, examples)


def export_weak_samples(samples: list[dict[str, Any]], out_dir: Path, examples: list[dict[str, Any]]) -> None:
    """把 reward 最低的若干样本写成 SVG 文件，便于人工定性看短板。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 按 score 升序取最低 10
    ordered = sorted(samples, key=lambda s: s["score"])[:10]
    for i, s in enumerate(ordered):
        fname = out_dir / f"weak_{i:02d}_id{s['id']}_t{s['temperature']}_score{s['score']:.4f}.svg"
        fname.write_text(s["svg"], encoding="utf-8")


def read_chat_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到验证集: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_message(example: dict[str, Any], role: str) -> str:
    for message in example.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def read_predictions(path: Path, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_by_id = {index: get_message(example, "user") for index, example in enumerate(examples)}
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = item.get("id", index)
            rows.append(
                {
                    "id": item_id,
                    "prompt": item.get("prompt", prompt_by_id.get(int(item_id), "")),
                    "svg": item.get("svg") or item.get("prediction") or item.get("content", ""),
                    "source": item.get("source", "prediction_file"),
                }
            )
    return rows


def generate_predictions(args: argparse.Namespace, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在有 transformers/peft 的环境中生成 SVG；本机无模型时会明确报错。"""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "当前环境缺少 transformers/torch；请在 AI Studio 或安装依赖后运行，"
            "或使用 --mode reference / --predictions-jsonl 仅评估已有 SVG。"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_dtype = resolve_generation_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=load_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("加载 LoRA adapter 需要安装 peft。") from exc
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    predictions = []
    for index, example in enumerate(examples):
        messages = [message for message in example["messages"] if message["role"] != "assistant"]
        prompt_text = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        predictions.append(
            {
                "id": index,
                "prompt": get_message(example, "user"),
                "svg": extract_svg(generated),
                "source": "base" if not args.adapter_path else "adapter",
            }
        )

    args.generations_path.parent.mkdir(parents=True, exist_ok=True)
    with args.generations_path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return predictions


def build_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """已迁移到 student_kit.prompt_utils.build_prompt，此处保留转发以兼容旧引用。"""
    return build_prompt_shared(tokenizer, messages)


def resolve_generation_dtype() -> Any:
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    # Gemma 3 270M 在本机 FP16 loss/采样上可能产生 NaN，Ada GPU 支持 BF16 时优先用它。
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


if __name__ == "__main__":
    main()
