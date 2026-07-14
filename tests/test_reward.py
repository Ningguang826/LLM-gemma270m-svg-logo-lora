import unittest

from student_kit.reward import score_svg


class RewardTests(unittest.TestCase):
    def test_valid_simple_svg_scores_high(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<circle cx="128" cy="128" r="80" fill="#1B3A5C"/>'
            '<path d="M80 160 C120 100 150 100 180 160" fill="none" stroke="#5DA88E" stroke-width="8"/>'
            '<rect x="112" y="72" width="32" height="88" rx="8" fill="#F2A93B"/>'
            "</svg>"
        )
        result = score_svg(svg, "circular badge with teal curve and golden rectangle")
        self.assertGreater(result.score, 0.75)
        self.assertEqual(result.subscores["valid_xml"], 1.0)
        self.assertEqual(result.subscores["safe_svg"], 1.0)
        self.assertEqual(result.subscores["viewbox"], 1.0)

    def test_invalid_xml_scores_low(self):
        result = score_svg("<svg><circle></svg>", "circle")
        self.assertLess(result.score, 0.25)
        self.assertIn("xml_parse_error", result.reasons)

    def test_script_is_penalized(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<script>bad()</script><circle cx="128" cy="128" r="80" fill="red"/>'
            "</svg>"
        )
        result = score_svg(svg, "red circle")
        self.assertLess(result.subscores["safe_svg"], 1.0)
        self.assertIn("blocked_tag:script", result.reasons)

    def test_out_of_canvas_is_penalized(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<circle cx="9999" cy="9999" r="80" fill="red"/>'
            '<circle cx="128" cy="128" r="20" fill="blue"/>'
            "</svg>"
        )
        result = score_svg(svg, "red and blue circles")
        self.assertLess(result.subscores["geometry"], 0.7)
        self.assertIn("extreme_coordinate", result.reasons)

    def test_prompt_alignment_keyword_match(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<polygon points="128,40 200,180 56,180" fill="#5A7"/>'
            '<circle cx="128" cy="90" r="20" fill="#FC0"/>'
            "</svg>"
        )
        result = score_svg(svg, "a mountain peak under the sun")
        self.assertGreater(result.subscores["prompt_alignment"], 0.6)

    def test_prompt_alignment_low_coverage(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<rect x="40" y="40" width="176" height="176" fill="#1B3A5C"/>'
            '<rect x="60" y="60" width="40" height="40" fill="#1B3A5C"/>'
            "</svg>"
        )
        result = score_svg(svg, "a mountain peak under the sun with leaves")
        self.assertLess(result.subscores["prompt_alignment"], 0.55)
        self.assertIn("low_prompt_keyword_coverage", result.reasons)

    def test_partial_subscore_cannot_be_clipped_to_full_reward(self):
        """权重归一化后，提示词对齐不足不能被其他满分项截断为总分 1。"""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<rect x="20" y="20" width="216" height="216" fill="#1B3A5C"/>'
            '<circle cx="90" cy="128" r="42" fill="#F2A93B"/>'
            '<circle cx="166" cy="128" r="42" fill="#5DA88E"/>'
            "</svg>"
        )
        result = score_svg(svg, "a five-point star icon")
        self.assertLess(result.subscores["prompt_alignment"], 1.0)
        self.assertLess(result.score, 1.0)

    def test_repetitive_shapes_detected(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<circle cx="40" cy="40" r="10" fill="#000"/>'
            '<circle cx="40" cy="40" r="10" fill="#000"/>'
            '<circle cx="40" cy="40" r="10" fill="#000"/>'
            '<circle cx="40" cy="40" r="10" fill="#000"/>'
            '<circle cx="40" cy="40" r="10" fill="#000"/>'
            "</svg>"
        )
        result = score_svg(svg, "dots")
        self.assertLess(result.subscores["non_degenerate"], 0.75)
        self.assertIn("repetitive_shapes", result.reasons)

    def test_palette_prompt_color_hit(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<rect x="20" y="20" width="216" height="216" fill="#1F4E79"/>'
            '<circle cx="128" cy="128" r="60" fill="#F2994A"/>'
            "</svg>"
        )
        result = score_svg(svg, "use colors #1F4E79 and #F2994A")
        self.assertEqual(result.subscores["palette"], 1.0)

    def test_palette_prompt_color_miss(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<rect x="20" y="20" width="216" height="216" fill="#111111"/>'
            '<circle cx="128" cy="128" r="60" fill="#EEEEEE"/>'
            "</svg>"
        )
        result = score_svg(svg, "use colors #1F4E79 and #F2994A")
        self.assertLess(result.subscores["palette"], 1.0)
        self.assertIn("missed_prompt_palette", result.reasons)

    def test_structure_reasonable_shape_count(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
            '<circle cx="50" cy="50" r="20" fill="#1B3A5C"/>'
            '<circle cx="128" cy="128" r="20" fill="#5DA88E"/>'
            '<circle cx="206" cy="206" r="20" fill="#F2A93B"/>'
            "</svg>"
        )
        result = score_svg(svg, "three dots")
        self.assertGreaterEqual(result.subscores["structure"], 0.8)

    def test_structure_too_many_shapes(self):
        circles = "".join(
            f'<circle cx="{20 + (i % 10) * 22}" cy="{20 + (i // 10) * 22}" r="6" fill="#1B3A5C"/>'
            for i in range(45)
        )
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">{circles}</svg>'
        result = score_svg(svg, "many dots")
        self.assertIn("too_many_shapes", result.reasons)
        self.assertLessEqual(result.subscores["structure"], 0.8)


if __name__ == "__main__":
    unittest.main()
