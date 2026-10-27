#!/usr/bin/env python3
"""Tests for publish_index_update.sh.

The script pushes to origin. Tests give it a local bare repository so they
never touch the real remote. They check that a configuration page is applied
on top of a main branch that moved, and that the root index is rebuilt from
that result.
"""
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import generate_root_index


REPO = Path(__file__).resolve().parent
SCRIPT = REPO / "publish_index_update.sh"
CU124_CP310 = "nvdiffrec_render-0.0.1+cu124torch25-cp310-cp310-linux_x86_64.whl"
CU124_CP311 = "nvdiffrec_render-0.0.1+cu124torch25-cp311-cp311-linux_x86_64.whl"
CU128_CP310 = "nvdiffrec_render-0.0.1+cu128torch29-cp310-cp310-linux_x86_64.whl"
CU128_CP311 = "nvdiffrec_render-0.0.1+cu128torch29-cp311-cp311-linux_x86_64.whl"


def config_page(*filenames):
    lines = [
        "<!DOCTYPE html>",
        "<html><body>",
    ]
    for name in filenames:
        lines.append(f'<a href="https://example.com/{name}">{name}</a><br>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        text=True,
        capture_output=True,
    )


def configure_identity(repo):
    git(repo, "config", "user.name", "index-test")
    git(repo, "config", "user.email", "index-test@example.com")


class PublishIndexUpdateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.bare = root / "origin.git"
        self.seed = root / "seed"
        self.work = root / "work"
        git(root, "init", "--bare", "-b", "main", str(self.bare))
        git(root, "clone", str(self.bare), str(self.seed))
        configure_identity(self.seed)
        self._link_generators(self.seed)
        self._write_pages(self.seed, (CU124_CP310,), (CU128_CP310,))
        self._commit_and_push(self.seed, "initial indexes")
        git(root, "clone", str(self.bare), str(self.work))
        configure_identity(self.work)
        self._link_generators(self.work)

    def _link_generators(self, repo):
        for name in ("generate_root_index.py", "validate_index.py"):
            target = repo / name
            if not target.exists():
                target.symlink_to(REPO / name)

    def _write_pages(self, repo, cu124_names, cu128_names):
        cu124 = repo / "wheels" / "cu124-torch251"
        cu128 = repo / "wheels" / "cu128-torch291"
        cu124.mkdir(parents=True, exist_ok=True)
        cu128.mkdir(parents=True, exist_ok=True)
        (cu124 / "index.html").write_text(config_page(*cu124_names), encoding="utf-8")
        (cu128 / "index.html").write_text(config_page(*cu128_names), encoding="utf-8")
        subprocess.run(["python3", "generate_root_index.py"], cwd=repo, check=True, text=True)

    def _commit_and_push(self, repo, message):
        git(repo, "add", "wheels")
        git(repo, "commit", "-m", message)
        git(repo, "push", "origin", "main")

    def _run(self, repo, env_extra, check=True):
        env = os.environ.copy()
        env.update(env_extra)
        return subprocess.run(
            [str(SCRIPT)],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=check,
        )

    def test_reapplies_the_page_onto_main_and_rebuilds_the_root_index(self):
        self._write_pages(self.seed, (CU124_CP310, CU124_CP311), (CU128_CP310,))
        self._commit_and_push(self.seed, "cu124 gained cp311")
        cu128 = self.work / "wheels" / "cu128-torch291" / "index.html"
        cu128.write_text(config_page(CU128_CP310, CU128_CP311), encoding="utf-8")

        result = self._run(
            self.work,
            {
                "CUDA_VERSION": "12.8",
                "PYTORCH_VERSION": "2.9.1",
                "CUDA_SHORT": "128",
                "TORCH_SHORT": "291",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Pushed index update", result.stdout)

        check = Path(self.tmp.name) / "check"
        git(self.bare.parent, "clone", str(self.bare), str(check))
        cu124_html = (check / "wheels" / "cu124-torch251" / "index.html").read_text(encoding="utf-8")
        cu128_html = (check / "wheels" / "cu128-torch291" / "index.html").read_text(encoding="utf-8")
        self.assertIn(CU124_CP311, cu124_html)
        self.assertIn(CU128_CP311, cu128_html)
        root_html = (check / "wheels" / "index.html").read_text(encoding="utf-8")
        self.assertEqual(root_html, generate_root_index.build_html(check / "wheels"))
        self.assertIn(CU124_CP311, root_html)
        self.assertIn(CU128_CP311, root_html)
        subject = git(check, "log", "-1", "--format=%s").stdout.strip()
        self.assertEqual(subject, "Update index: CUDA 12.8 + PyTorch 2.9.1")

    def test_unchanged_index_does_not_commit(self):
        before = git(self.bare, "rev-parse", "main").stdout.strip()
        result = self._run(
            self.work,
            {
                "CUDA_VERSION": "12.8",
                "PYTORCH_VERSION": "2.9.1",
                "CUDA_SHORT": "128",
                "TORCH_SHORT": "291",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No index changes", result.stdout)
        after = git(self.bare, "rev-parse", "main").stdout.strip()
        self.assertEqual(before, after)

    def test_rejected_push_is_retried(self):
        hook = self.bare / "hooks" / "pre-receive"
        hook.write_text(
            "#!/bin/sh\n"
            'count="$GIT_DIR/reject-count"\n'
            "n=0\n"
            'if [ -f "$count" ]; then n=$(cat "$count"); fi\n'
            "n=$((n + 1))\n"
            'echo "$n" > "$count"\n'
            'if [ "$n" -eq 1 ]; then\n'
            '  echo "reject first push" >&2\n'
            "  exit 1\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)
        cu128 = self.work / "wheels" / "cu128-torch291" / "index.html"
        cu128.write_text(config_page(CU128_CP310, CU128_CP311), encoding="utf-8")
        result = self._run(
            self.work,
            {
                "CUDA_VERSION": "12.8",
                "PYTORCH_VERSION": "2.9.1",
                "CUDA_SHORT": "128",
                "TORCH_SHORT": "291",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Push attempt 1 failed", result.stderr)
        self.assertIn("Pushed index update", result.stdout)
        self.assertEqual((self.bare / "reject-count").read_text(encoding="utf-8").strip(), "2")
        check = Path(self.tmp.name) / "check-retry"
        git(self.bare.parent, "clone", str(self.bare), str(check))
        # The rejected commit is discarded on retry, so main gains one commit.
        self.assertEqual(git(check, "rev-list", "--count", "main").stdout.strip(), "2")

    def test_missing_config_page_fails(self):
        (self.work / "wheels" / "cu128-torch291" / "index.html").unlink()
        result = self._run(
            self.work,
            {
                "CUDA_VERSION": "12.8",
                "PYTORCH_VERSION": "2.9.1",
                "CUDA_SHORT": "128",
                "TORCH_SHORT": "291",
            },
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
