#!/usr/bin/env python3
"""Tests for generate_root_index.py.

Directory names drop the dots in the version. PyTorch 2.15.0 is therefore
cu130-torch2150, four digits, and the old three-digit split rejects it.
"""
import tempfile
import unittest
from pathlib import Path

import generate_root_index


ROOT = Path(__file__).resolve().parent
WHEEL_215 = "nvdiffrec_render-0.0.1+cu130torch215-cp310-cp310-linux_x86_64.whl"
WHEEL_291 = "nvdiffrec_render-0.0.1+cu128torch29-cp311-cp311-linux_x86_64.whl"
DOUBLED = "nvdiffrec_render-0.0.1+cu130torch215-cp310+cu130torch215-cp310-win_amd64.whl"
DIGEST = "ab" * 32


def config_page(*hrefs):
    lines = ["<!DOCTYPE html>", "<html><body>"]
    for href, label in hrefs:
        lines.append(f'<a href="{href}">{label}</a><br>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


class VersionsFromDirnameTest(unittest.TestCase):
    def test_three_digit_torch_tag(self):
        self.assertEqual(
            generate_root_index.versions_from_dirname("cu128-torch291"),
            ("12.8", "2.9.1"),
        )
        self.assertEqual(
            generate_root_index.versions_from_dirname("cu124-torch251"),
            ("12.4", "2.5.1"),
        )

    def test_four_digit_torch_tag_is_a_two_digit_minor(self):
        self.assertEqual(generate_root_index.torch_from_short("2150"), "2.15.0")
        self.assertEqual(generate_root_index.torch_from_short("2100"), "2.10.0")
        self.assertEqual(generate_root_index.torch_from_short("2151"), "2.15.1")
        self.assertEqual(
            generate_root_index.versions_from_dirname("cu130-torch2150"),
            ("13.0", "2.15.0"),
        )
        self.assertEqual(
            generate_root_index.versions_from_dirname("cu1210-torch2150"),
            ("12.10", "2.15.0"),
        )

    def test_ambiguous_tags_are_rejected(self):
        self.assertIsNone(generate_root_index.torch_from_short("29"))
        self.assertIsNone(generate_root_index.torch_from_short("21510"))
        self.assertIsNone(generate_root_index.versions_from_dirname("cu128-torch29"))
        self.assertIsNone(generate_root_index.versions_from_dirname("cu12-torch291"))
        self.assertIsNone(generate_root_index.versions_from_dirname("not-a-config"))


class BuildHtmlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wheels = Path(self.tmp.name)

    def write_config(self, name, html):
        directory = self.wheels / name
        directory.mkdir()
        (directory / "index.html").write_text(html, encoding="utf-8")

    def test_two_digit_minor_is_listed_and_keeps_the_hash_fragment(self):
        href = f"https://example.com/releases/{WHEEL_215}#sha256={DIGEST}"
        self.write_config(
            "cu130-torch2150",
            config_page((href, WHEEL_215), (f"https://example.com/{DOUBLED}", DOUBLED)),
        )
        html = generate_root_index.build_html(self.wheels)
        self.assertIn("<td>13.0</td><td>2.15.0</td><td>3.10-3.10</td>", html)
        self.assertIn(f'href="{href}"', html)
        self.assertNotIn(DOUBLED, html)
        self.assertIn("<code>0.0.1+cu130torch215</code>", html)

    def test_checked_in_index_is_unchanged_by_the_wider_parser(self):
        generated = generate_root_index.build_html(ROOT / "wheels")
        checked_in = (ROOT / "wheels" / "index.html").read_text(encoding="utf-8")
        self.assertEqual(generated, checked_in)
        self.assertIn("cu128-torch291", generated)
        self.assertNotIn("2.15.0", generated)

    def test_unrecognized_directory_still_fails(self):
        self.write_config("notes", "<html><body></body></html>")
        with self.assertRaises(SystemExit):
            generate_root_index.build_html(self.wheels)

    def test_both_tag_widths_are_sorted_into_the_table(self):
        self.write_config(
            "cu128-torch291",
            config_page((f"https://example.com/{WHEEL_291}", WHEEL_291)),
        )
        self.write_config(
            "cu130-torch2150",
            config_page((f"https://example.com/{WHEEL_215}", WHEEL_215)),
        )
        html = generate_root_index.build_html(self.wheels)
        self.assertLess(html.index("12.8"), html.index("13.0"))
        self.assertIn("2.9.1", html)
        self.assertIn("2.15.0", html)


if __name__ == "__main__":
    unittest.main()
