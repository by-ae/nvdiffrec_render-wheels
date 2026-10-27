#!/usr/bin/env python3
"""Check the README and root index against wheels the repo actually publishes.

The supported-configuration table and the pip install commands have to match
the wheel links in wheels/<config>/index.html and the Python/platform matrix
in the corresponding build workflow. Configurations with an empty release
index must not be offered as an install command.
"""
import re
import unittest
from pathlib import Path

import generate_root_index
import validate_index


ROOT = Path(__file__).resolve().parent
WHEELS = ROOT / "wheels"
WORKFLOWS = ROOT / ".github" / "workflows"
README = ROOT / "README.md"


def parse_workflow(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    def env_value(name):
        match = re.search(rf"(?m)^  {name}: '([^']+)'", text)
        if not match:
            raise AssertionError(f"{path.name} missing {name}")
        return match.group(1)

    def job_pythons(job_name):
        match = re.search(
            rf"(?ms)^  {job_name}:\n(.*?)(?=^  [A-Za-z0-9-]+:\n|\Z)",
            text,
        )
        if not match:
            return []
        versions = re.search(r"python:\s*\[([^\]]+)\]", match.group(1))
        if not versions:
            raise AssertionError(f"{path.name} {job_name} has no python matrix")
        return [part.strip().strip("'\"") for part in versions.group(1).split(",")]

    index_urls = re.findall(
        r"https://download\.pytorch\.org/whl/cu\d+",
        text,
    )
    return {
        "cuda": env_value("CUDA_VERSION"),
        "torch": env_value("PYTORCH_VERSION"),
        "name": f"cu{env_value('CUDA_SHORT')}-torch{env_value('TORCH_SHORT')}",
        "linux": job_pythons("build-linux"),
        "windows": job_pythons("build-windows"),
        "index_urls": index_urls,
    }


def wheels_by_platform(config_name: str):
    html = (WHEELS / config_name / "index.html").read_text(encoding="utf-8")
    linux = set()
    windows = set()
    for filename, _href in generate_root_index.valid_wheel_links(html):
        match = re.search(r"-cp(3\d+)-cp\d+-(linux_x86_64|win_amd64)\.whl$", filename)
        if not match:
            raise AssertionError(f"Unexpected wheel filename: {filename}")
        tag, platform = match.groups()
        version = f"{tag[0]}.{tag[1:]}"
        if platform == "linux_x86_64":
            linux.add(version)
        else:
            windows.add(version)
    return linux, windows


def readme_table(readme: str):
    section = readme.split("## Supported Configurations", 1)[1].split("## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in {"CUDA", "------"} or set(cells[0]) <= {"-"}:
            continue
        if not re.fullmatch(r"[0-9.]+", cells[0]):
            continue
        rows.append(
            {
                "cuda": cells[0],
                "torch": cells[1],
                "pythons": [part.strip() for part in cells[2].split(",")],
                "platforms": cells[3],
            }
        )
    return rows


def install_configs(readme: str):
    return re.findall(
        r"--find-links https://by-ae\.github\.io/nvdiffrec_render-wheels/([a-z0-9-]+)/",
        readme,
    )


class ReadmeCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = README.read_text(encoding="utf-8")
        cls.workflows = [
            parse_workflow(path)
            for path in sorted(WORKFLOWS.glob("build-cu*-torch*.yml"))
        ]
        cls.by_name = {item["name"]: item for item in cls.workflows}

    def test_root_index_matches_the_generator(self):
        generated = generate_root_index.build_html(WHEELS)
        checked_in = (WHEELS / "index.html").read_text(encoding="utf-8")
        self.assertEqual(checked_in, generated)

    def test_published_rows_match_index_and_workflows(self):
        rows = {(row["cuda"], row["torch"]): row for row in readme_table(self.readme)}
        offered = install_configs(self.readme)
        self.assertEqual(len(offered), len(set(offered)))

        for workflow in self.workflows:
            linux, windows = wheels_by_platform(workflow["name"])
            key = (workflow["cuda"], workflow["torch"])
            if not linux and not windows:
                self.assertNotIn(key, rows, workflow["name"])
                self.assertNotIn(workflow["name"], offered)
                continue

            self.assertEqual(set(workflow["linux"]), linux, workflow["name"])
            self.assertEqual(set(workflow["windows"]), windows, workflow["name"])
            self.assertIn(workflow["name"], offered)
            row = rows[key]
            pythons = sorted(linux | windows, key=lambda version: [int(part) for part in version.split(".")])
            self.assertEqual(row["pythons"], pythons)
            if linux and windows:
                self.assertEqual(row["platforms"], "Linux, Windows")
            elif linux:
                self.assertEqual(row["platforms"], "Linux only")
            else:
                self.assertEqual(row["platforms"], "Windows only")
            short = workflow["name"].split("-", 1)[0]  # cu124
            self.assertTrue(workflow["index_urls"], workflow["name"])
            self.assertTrue(
                all(url.endswith("/" + short) for url in workflow["index_urls"]),
                workflow["index_urls"],
            )

        published_keys = {
            (workflow["cuda"], workflow["torch"])
            for workflow in self.workflows
            if any(wheels_by_platform(workflow["name"]))
        }
        self.assertEqual(set(rows), published_keys)
        published_names = {
            workflow["name"]
            for workflow in self.workflows
            if (workflow["cuda"], workflow["torch"]) in published_keys
        }
        self.assertEqual(set(offered), published_names)

    def test_build_jobs_validate_the_wheel_archive(self):
        """The import check used to run inside nvdiffrec_render_pkg.

        Python then loads the source tree, which has no _C extension, and
        the job fails after a good wheel was built. Run 40233374869 failed
        this way on every matrix leg and skipped the release.
        """
        command = "python validate_wheel.py nvdiffrec_render_pkg/dist/*.whl"
        for path in sorted(WORKFLOWS.glob("build-cu*-torch*.yml")):
            text = path.read_text(encoding="utf-8")
            expected = 2 if "\n  build-windows:\n" in text else 1
            self.assertEqual(text.count(command), expected, path.name)
            self.assertNotIn("import nvdiffrec_render", text, path.name)

    def test_release_jobs_upload_wheels_and_index_updates_are_serialized(self):
        for path in sorted(WORKFLOWS.glob("build-cu*-torch*.yml")):
            text = path.read_text(encoding="utf-8")
            self.assertIn("\n          files: wheels/*.whl\n", text, path.name)
            self.assertIn("fail_on_unmatched_files: true", text)
            self.assertIn("group: wheel-index-update", text)
            self.assertIn("queue: max", text)
            self.assertIn("cancel-in-progress: false", text)
            self.assertNotIn("git pull --rebase origin main || true", text)
            self.assertNotIn("git rebase --abort", text, path.name)
            self.assertIn("\n        run: ./publish_index_update.sh\n", text, path.name)
            self.assertNotIn("- '.github/workflows/build-cu*-torch*.yml'", text)

    def test_cu130_is_described_as_unpublished(self):
        workflow = self.by_name["cu130-torch291"]
        self.assertEqual(workflow["linux"], ["3.10", "3.11", "3.12", "3.13"])
        self.assertEqual(workflow["windows"], [])
        self.assertIn("https://download.pytorch.org/whl/cu130", workflow["index_urls"])
        linux, windows = wheels_by_platform("cu130-torch291")
        self.assertEqual(linux, set())
        self.assertEqual(windows, set())
        unpublished = self.readme.split("## Not published yet", 1)[1].split("## ", 1)[0]
        self.assertIn("cu130-torch291", unpublished)
        self.assertIn("build-cu130-torch291.yml", unpublished)
        self.assertIn("https://download.pytorch.org/whl/cu130", unpublished)
        self.assertNotIn("--find-links", unpublished)
        # The page URL may be mentioned, but not as an install command.
        self.assertNotIn(
            "--find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu130-torch291/",
            self.readme,
        )

    def test_generator_skips_invalid_wheel_links_and_rejects_gaps(self):
        html = """
        <a href="nvdiffrec_render-0.0.1+cu124torch25-cp310-cp310-linux_x86_64.whl">a</a>
        <a href="nvdiffrec_render-0.0.1+cu124torch25-cp310+cu124torch25-cp310-win_amd64.whl">bad</a>
        <a href="nvdiffrec_render-0.0.1+cu124torch25-cp312-cp312-win_amd64.whl">c</a>
        """
        names = [name for name, _href in generate_root_index.valid_wheel_links(html)]
        self.assertEqual(
            names,
            [
                "nvdiffrec_render-0.0.1+cu124torch25-cp310-cp310-linux_x86_64.whl",
                "nvdiffrec_render-0.0.1+cu124torch25-cp312-cp312-win_amd64.whl",
            ],
        )
        with self.assertRaises(SystemExit):
            generate_root_index.python_range(names)

    def test_checked_in_indexes_have_no_invalid_wheel_links(self):
        errors = []
        for path in sorted(WHEELS.rglob("index.html")):
            parser = validate_index.IndexHTMLParser()
            parser.feed(path.read_text(encoding="utf-8"))
            for href in parser.links:
                message = validate_index.wheel_href_error(href)
                if message:
                    errors.append(f"{path}: {message}")
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
