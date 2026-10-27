#!/usr/bin/env python3
"""Tests for generate_config_index.sh.

The script shells out to `gh` for release assets. Tests put a fake `gh` on
PATH so they check PEP 503 links, sha256 fragments, and which assets are
omitted without calling GitHub.
"""
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT = REPO_ROOT / "generate_config_index.sh"
VALID_LINUX = "nvdiffrec_render-0.0.1+cu126torch26-cp311-cp311-linux_x86_64.whl"
VALID_WIN = "nvdiffrec_render-0.0.1+cu126torch26-cp310-cp310-win_amd64.whl"
DOUBLED = "nvdiffrec_render-0.0.1+cu126torch26-cp310+cu126torch26-cp310-win_amd64.whl"
LINUX_HASH = "b" * 64
WIN_HASH = "a" * 64


def write_fake_gh(bin_dir: Path, body: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / "gh"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class GenerateConfigIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.bin_dir = self.work / "bin"

    def env(self, **extra):
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"GH_TOKEN", "GITHUB_TOKEN", "GITHUB_ACTIONS", "GITHUB_REPOSITORY"}
        }
        env["PATH"] = str(self.bin_dir) + os.pathsep + env.get("PATH", "")
        env.update(extra)
        return env

    def run_script(self, args, env, check=False):
        return subprocess.run(
            [str(SCRIPT), *args],
            cwd=self.work,
            env=env,
            text=True,
            capture_output=True,
            check=check,
        )

    def test_usage_requires_four_arguments(self):
        result = self.run_script([], self.env())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stderr)

    def test_short_tags_must_match_versions(self):
        result = self.run_script(["12.6", "2.6.0", "128", "260"], self.env())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)
        self.assertFalse((self.work / "wheels").exists())

    def test_actions_without_a_token_does_not_clobber_an_existing_index(self):
        index = self.work / "wheels" / "cu126-torch260" / "index.html"
        index.parent.mkdir(parents=True)
        index.write_text("SENTINEL", encoding="utf-8")
        result = self.run_script(
            ["12.6", "2.6.0", "126", "260"],
            self.env(GITHUB_ACTIONS="true"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(index.read_text(encoding="utf-8"), "SENTINEL")
        self.assertIn("GH_TOKEN", result.stderr)

    def test_without_token_writes_a_placeholder_and_no_wheel_links(self):
        result = self.run_script(["12.6", "2.6.0", "126", "260"], self.env())
        self.assertEqual(result.returncode, 0, result.stderr)
        html = (self.work / "wheels" / "cu126-torch260" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Wheel links will be added by the workflow", html)
        self.assertNotIn(".whl", html)
        self.assertIn(
            "https://by-ae.github.io/nvdiffrec_render-wheels/cu126-torch260/",
            html,
        )
        self.assertIn("CUDA 12.6 + PyTorch 2.6.0", html)

    def test_release_links_are_pep503_and_skip_invalid_assets(self):
        write_fake_gh(
            self.bin_dir,
            """#!/bin/sh
printf '%s\\n' "$*" > "$FAKE_GH_LOG"
if [ "$1" != "release" ] || [ "$2" != "view" ] || [ "$3" != "cu126-torch260" ]; then
    echo "unexpected args: $*" >&2
    exit 2
fi
case "$*" in
    *"--repo example/wheels"*) ;;
    *) echo "missing repo: $*" >&2; exit 3 ;;
esac
case "$*" in
    *"--json assets"*) ;;
    *) echo "missing json: $*" >&2; exit 4 ;;
esac
printf '%s\\t%s\\n' \\
    '{doubled}' 'sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc' \\
    '{linux}' 'sha256:{linux_hash}' \\
    'notes.txt' 'sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd' \\
    '{win}' 'sha256:{win_hash}' \\
    'nvdiffrec_render-0.0.1+cu126torch26-cp312-cp312-macosx_11_0_arm64.whl' 'sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' \\
    '{linux}' 'sha256:GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG'
""".format(
                doubled=DOUBLED,
                linux=VALID_LINUX,
                linux_hash=LINUX_HASH,
                win=VALID_WIN,
                win_hash=WIN_HASH,
            ),
        )
        log = self.work / "gh.log"
        result = self.run_script(
            ["12.6", "2.6.0", "126", "260"],
            self.env(
                GH_TOKEN="test-token",
                GITHUB_REPOSITORY="example/wheels",
                FAKE_GH_LOG=str(log),
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Skipping", result.stderr)
        self.assertIn(DOUBLED, result.stderr)
        html = (self.work / "wheels" / "cu126-torch260" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn(DOUBLED, html)
        self.assertNotIn("notes.txt", html)
        self.assertNotIn("macosx", html)
        self.assertNotIn("sha256=GGGG", html)
        linux_href = (
            "https://github.com/example/wheels/releases/download/cu126-torch260/"
            f"{VALID_LINUX}#sha256={LINUX_HASH}"
        )
        win_href = (
            "https://github.com/example/wheels/releases/download/cu126-torch260/"
            f"{VALID_WIN}#sha256={WIN_HASH}"
        )
        self.assertIn(f'<a href="{linux_href}">{VALID_LINUX}</a>', html)
        self.assertIn(f'<a href="{win_href}">{VALID_WIN}</a>', html)
        # cp310 sorts before cp311
        self.assertLess(html.index(VALID_WIN), html.index(VALID_LINUX))
        self.assertIn("</html>", html)
        self.assertNotIn("No wheel files are published", html)

    def test_empty_release_keeps_the_previous_index_when_gh_fails(self):
        index = self.work / "wheels" / "cu126-torch260" / "index.html"
        index.parent.mkdir(parents=True)
        index.write_text("PREVIOUS", encoding="utf-8")
        write_fake_gh(self.bin_dir, "#!/bin/sh\nexit 1\n")
        result = self.run_script(
            ["12.6", "2.6.0", "126", "260"],
            self.env(GH_TOKEN="test-token", GITHUB_REPOSITORY="example/wheels"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(index.read_text(encoding="utf-8"), "PREVIOUS")

    def test_empty_asset_list_says_nothing_is_published(self):
        write_fake_gh(
            self.bin_dir,
            """#!/bin/sh
exit 0
""",
        )
        result = self.run_script(
            ["13.0", "2.9.1", "130", "291"],
            self.env(GH_TOKEN="test-token", GITHUB_REPOSITORY="by-ae/nvdiffrec_render-wheels"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        html = (self.work / "wheels" / "cu130-torch291" / "index.html").read_text(encoding="utf-8")
        self.assertIn("No wheel files are published for this release yet.", html)
        self.assertNotIn(".whl", html)
        self.assertIn("cu130-torch291/", html)


if __name__ == "__main__":
    unittest.main()
