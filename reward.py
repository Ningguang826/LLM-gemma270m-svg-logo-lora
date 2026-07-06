"""对外入口：复用 `student_kit.reward` 的实现。"""

from student_kit.reward import aggregate_scores, extract_svg, reward, score, score_batch, score_svg

__all__ = ["aggregate_scores", "extract_svg", "reward", "score", "score_batch", "score_svg"]
