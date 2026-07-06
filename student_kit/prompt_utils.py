"""训练/评估共用的 prompt 构造。

`train_peft.py` 和 `eval_self.py` 必须使用同一个 prompt 格式，否则 reward
评估会因 prompt 不匹配失准。这里抽出唯一实现，两边 import。
"""

from __future__ import annotations

from typing import Any


def build_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """构造喂给模型的纯文本 prompt。

    若 tokenizer 自带 chat_template，优先使用官方模板（评估时）；训练侧
    显式传入无 chat_template 的 tokenizer，走纯文本分支。
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    user = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
    return (
        f"{tokenizer.bos_token or ''}"
        f"{system}\n\n"
        f"Description:\n{user}\n\n"
        "Output exactly one complete SVG document and nothing else:\n"
    )
