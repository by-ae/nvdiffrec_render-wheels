#!/usr/bin/env python3
"""
Unit tests for find_wheel.py.
Tests version parsing, config discovery, matching logic, and integration scenarios.
"""
import io
import subprocess
import unittest
import tempfile
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import find_wheel
sys.path.insert(0, str(Path(__file__).parent))
import find_wheel


class TestVersionParsing(unittest.TestCase):
    """Test version string parsing functions."""
    
    def test_parse_torch_version_standard(self):
        """Test parsing standard PyTorch version strings."""
        self.assertEqual(find_wheel.parse_version('2.9.1'), (2, 9, 1))
        self.assertEqual(find_wheel.parse_version('2.8.0'), (2, 8, 0))
        self.assertEqual(find_wheel.parse_version('2.6.0'), (2, 6, 0))
        self.assertEqual(find_wheel.parse_version('2.5.1'), (2, 5, 1))
    
    def test_parse_torch_version_with_suffix(self):
        """Test parsing PyTorch versions with local suffixes."""
        self.assertEqual(find_wheel.parse_version('2.9.1+cu128'), (2, 9, 1))
        self.assertEqual(find_wheel.parse_version('2.8.0+cu126'), (2, 8, 0))
        self.assertEqual(find_wheel.parse_version('2.5.1+cpu'), (2, 5, 1))
    
    def test_parse_cuda_version_standard(self):
        """Test parsing standard CUDA version strings."""
        self.assertEqual(find_wheel.parse_cuda_version('12.8'), (12, 8))
        self.assertEqual(find_wheel.parse_cuda_version('13.0'), (13, 0))
        self.assertEqual(find_wheel.parse_cuda_version('12.6'), (12, 6))
        self.assertEqual(find_wheel.parse_cuda_version('12.4'), (12, 4))
    
    def test_parse_cuda_version_short(self):
        """Test parsing short CUDA version strings."""
        self.assertEqual(find_wheel.parse_cuda_version('128'), (12, 8))
        self.assertEqual(find_wheel.parse_cuda_version('130'), (13, 0))
        self.assertEqual(find_wheel.parse_cuda_version('126'), (12, 6))
        self.assertEqual(find_wheel.parse_cuda_version('124'), (12, 4))
    
    def test_format_cuda_version(self):
        """Test formatting CUDA version tuples."""
        self.assertEqual(find_wheel.format_cuda_version((12, 8)), '12.8')
        self.assertEqual(find_wheel.format_cuda_version((13, 0)), '13.0')
        self.assertEqual(find_wheel.format_cuda_version((12, 6)), '12.6')
    
    def test_parse_invalid_version(self):
        """Test that invalid version strings raise ValueError."""
        with self.assertRaises(ValueError):
            find_wheel.parse_version('invalid')
        
        with self.assertRaises(ValueError):
            find_wheel.parse_version('2.9')
        
        with self.assertRaises(ValueError):
            find_wheel.parse_cuda_version('12')

    def test_parse_python_version(self):
        """A patch version is ignored; non-numeric text is an error."""
        self.assertEqual(find_wheel.parse_python_version('3.11'), (3, 11))
        self.assertEqual(find_wheel.parse_python_version('3.11.4'), (3, 11))
        with self.assertRaises(ValueError):
            find_wheel.parse_python_version('abc')
        with self.assertRaises(ValueError):
            find_wheel.parse_python_version('3')
        with self.assertRaises(ValueError):
            find_wheel.parse_python_version('3.x')


class TestConfigDiscovery(unittest.TestCase):
    """Test configuration discovery from wheels directory."""
    
    def setUp(self):
        """Create temporary directory for test configs."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
    
    def test_discover_real_configs(self):
        """Test discovery against the actual wheels directory."""
        wheels_dir = Path(__file__).parent / 'wheels'
        
        if not wheels_dir.exists():
            self.skipTest("wheels/ directory not found in repository")
        
        configs = find_wheel.discover_configs(wheels_dir)
        
        # Should find at least the 6 configs mentioned in the task
        self.assertGreater(len(configs), 0)
        
        # Check structure of returned configs
        for config in configs:
            self.assertIn('name', config)
            self.assertIn('cuda', config)
            self.assertIn('torch', config)
            
            # Validate types
            self.assertIsInstance(config['name'], str)
            self.assertIsInstance(config['cuda'], tuple)
            self.assertIsInstance(config['torch'], tuple)
            self.assertEqual(len(config['cuda']), 2)
            self.assertEqual(len(config['torch']), 3)
        
        # Check that configs are sorted (PyTorch desc, then CUDA desc)
        for i in range(len(configs) - 1):
            curr = configs[i]
            next_cfg = configs[i + 1]
            # Current should be >= next in sort order
            self.assertGreaterEqual(
                (curr['torch'], curr['cuda']),
                (next_cfg['torch'], next_cfg['cuda'])
            )
    
    def test_discover_mock_configs(self):
        """Test discovery with mock config directories."""
        wheels_dir = Path(self.test_dir)
        
        # Create mock config directories
        (wheels_dir / 'cu128-torch291').mkdir(parents=True)
        (wheels_dir / 'cu126-torch280').mkdir(parents=True)
        (wheels_dir / 'cu124-torch251').mkdir(parents=True)
        (wheels_dir / 'cu130-torch2150').mkdir(parents=True)
        
        configs = find_wheel.discover_configs(wheels_dir)
        
        self.assertEqual(len(configs), 4)
        
        # Check parsed values
        config_names = [c['name'] for c in configs]
        self.assertIn('cu128-torch291', config_names)
        self.assertIn('cu126-torch280', config_names)
        self.assertIn('cu124-torch251', config_names)
        self.assertIn('cu130-torch2150', config_names)
        
        # Check cu128-torch291 parsing
        cu128_config = next(c for c in configs if c['name'] == 'cu128-torch291')
        self.assertEqual(cu128_config['cuda'], (12, 8))
        self.assertEqual(cu128_config['torch'], (2, 9, 1))
        # 2.15.0 strips to four digits. Three-digit parsing would reject it.
        cu130_config = next(c for c in configs if c['name'] == 'cu130-torch2150')
        self.assertEqual(cu130_config['cuda'], (13, 0))
        self.assertEqual(cu130_config['torch'], (2, 15, 0))
    
    def test_discover_empty_directory(self):
        """Test discovery with empty wheels directory."""
        wheels_dir = Path(self.test_dir)
        wheels_dir.mkdir(exist_ok=True)
        
        configs = find_wheel.discover_configs(wheels_dir)
        
        self.assertEqual(len(configs), 0)
    
    def test_discover_nonexistent_directory(self):
        """Test discovery with non-existent directory."""
        wheels_dir = Path(self.test_dir) / 'nonexistent'
        
        configs = find_wheel.discover_configs(wheels_dir)
        
        self.assertEqual(len(configs), 0)
    
    def test_discover_ignores_invalid_names(self):
        """Test that invalid directory names are ignored."""
        wheels_dir = Path(self.test_dir)
        
        # Create valid and invalid directories
        (wheels_dir / 'cu128-torch291').mkdir(parents=True)
        (wheels_dir / 'invalid-name').mkdir(parents=True)
        (wheels_dir / '.git').mkdir(parents=True)
        (wheels_dir / 'README.md').touch()
        
        configs = find_wheel.discover_configs(wheels_dir)
        
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]['name'], 'cu128-torch291')

    def test_directory_scan_skips_an_index_with_no_wheel(self):
        """A config page with no wheel file is not an installable config."""
        wheels_dir = Path(self.test_dir)
        empty = wheels_dir / 'cu130-torch291'
        empty.mkdir(parents=True)
        (empty / 'index.html').write_text(
            '<h2>No wheels published yet</h2>'
            '<p>No wheel files are published for this release yet.</p>',
            encoding='utf-8',
        )
        (wheels_dir / 'cu128-torch291').mkdir()
        broken = wheels_dir / 'cu126-torch280'
        broken.mkdir()
        (broken / 'index.html').write_text(
            '<a href="nvdiffrec_render-0.0.1+cu126torch28-cp310+cu126torch28-cp310-cp310-linux_x86_64.whl">bad</a>',
            encoding='utf-8',
        )

        configs = find_wheel.discover_configs(wheels_dir)

        self.assertEqual([c['name'] for c in configs], ['cu128-torch291'])

    def test_directory_scan_reads_platforms_from_wheel_links(self):
        """A config index's wheel hrefs are the platform tags pip can install."""
        wheels_dir = Path(self.test_dir)
        config_dir = wheels_dir / 'cu128-torch291'
        config_dir.mkdir(parents=True)
        (config_dir / 'index.html').write_text(
            '<a href="nvdiffrec_render-0.0.1+cu128torch29-cp310-cp310-win_amd64.whl">w</a>',
            encoding='utf-8',
        )

        configs = find_wheel.discover_configs(wheels_dir)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]['platforms'], {'win_amd64'})


class TestConfigMatching(unittest.TestCase):
    """Test configuration matching logic."""
    
    def setUp(self):
        """Set up mock configs for testing."""
        self.configs = [
            {'name': 'cu130-torch291', 'cuda': (13, 0), 'torch': (2, 9, 1)},
            {'name': 'cu128-torch291', 'cuda': (12, 8), 'torch': (2, 9, 1)},
            {'name': 'cu128-torch280', 'cuda': (12, 8), 'torch': (2, 8, 0)},
            {'name': 'cu126-torch280', 'cuda': (12, 6), 'torch': (2, 8, 0)},
            {'name': 'cu126-torch260', 'cuda': (12, 6), 'torch': (2, 6, 0)},
            {'name': 'cu124-torch251', 'cuda': (12, 4), 'torch': (2, 5, 1)},
        ]
    
    def test_exact_match(self):
        """Test finding exact match."""
        exact, nearest = find_wheel.find_matching_config(
            python_version=(3, 11),
            torch_version=(2, 9, 1),
            cuda_version=(12, 8),
            configs=self.configs
        )
        
        self.assertIsNotNone(exact)
        self.assertEqual(exact['name'], 'cu128-torch291')
        
        # Exact match should also be the nearest
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest['name'], 'cu128-torch291')
    
    def test_no_exact_match_with_nearest(self):
        """Test nearest match when no exact match exists."""
        # Request torch 2.9.1 with CUDA 12.7 (not available)
        # torch (2,9,1) MM is (2,9)
        # configs with torch MM (2,9): cu130-torch291, cu128-torch291
        # filter CUDA <= 12.7: none (13.0 > 12.7, 12.8 > 12.7)
        # So nearest should be None
        exact, nearest = find_wheel.find_matching_config(
            python_version=(3, 11),
            torch_version=(2, 9, 1),
            cuda_version=(12, 7),
            configs=self.configs
        )
        
        self.assertIsNone(exact)
        self.assertIsNone(nearest)
    
    def test_nearest_match_lower_cuda(self):
        """Test nearest match suggests lower CUDA version."""
        # Request torch 2.8.0 with CUDA 12.7
        # Should suggest cu126-torch280 (CUDA 12.6 <= 12.7)
        exact, nearest = find_wheel.find_matching_config(
            python_version=(3, 11),
            torch_version=(2, 8, 0),
            cuda_version=(12, 7),
            configs=self.configs
        )
        
        self.assertIsNone(exact)
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest['name'], 'cu126-torch280')
    
    def test_nearest_match_prefers_highest_cuda(self):
        """Test that nearest match prefers highest compatible CUDA."""
        # Request torch 2.8.0 with CUDA 13.0
        # Should suggest cu128-torch280 (highest CUDA <= 13.0)
        exact, nearest = find_wheel.find_matching_config(
            python_version=(3, 11),
            torch_version=(2, 8, 0),
            cuda_version=(13, 0),
            configs=self.configs
        )
        
        self.assertIsNone(exact)
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest['name'], 'cu128-torch280')
    
    def test_no_match_different_torch_major_minor(self):
        """Test no match when torch major.minor doesn't match any config."""
        exact, nearest = find_wheel.find_matching_config(
            python_version=(3, 11),
            torch_version=(2, 7, 0),  # No 2.7.x configs
            cuda_version=(12, 8),
            configs=self.configs
        )
        
        self.assertIsNone(exact)
        self.assertIsNone(nearest)
    
    def test_none_inputs(self):
        """Test handling of None inputs."""
        exact, nearest = find_wheel.find_matching_config(
            python_version=(3, 11),
            torch_version=None,
            cuda_version=(12, 8),
            configs=self.configs
        )
        
        self.assertIsNone(exact)
        self.assertIsNone(nearest)
        
        exact, nearest = find_wheel.find_matching_config(
            python_version=(3, 11),
            torch_version=(2, 9, 1),
            cuda_version=None,
            configs=self.configs
        )
        
        self.assertIsNone(exact)
        self.assertIsNone(nearest)


class TestDetectEnvironment(unittest.TestCase):
    """Test environment detection."""
    
    @patch('sys.version_info', MagicMock(major=3, minor=11))
    def test_python_version_detection(self):
        """Test Python version is always detected."""
        python_ver, torch_ver, cuda_ver = find_wheel.detect_environment()
        
        self.assertEqual(python_ver, (3, 11))
    
    @patch('sys.version_info', MagicMock(major=3, minor=12))
    @patch('find_wheel.torch', None, create=True)
    def test_torch_not_installed(self):
        """Test detection when torch is not installed."""
        with patch.dict('sys.modules', {'torch': None}):
            python_ver, torch_ver, cuda_ver = find_wheel.detect_environment()
            
            self.assertEqual(python_ver, (3, 12))
            self.assertIsNone(torch_ver)
            self.assertIsNone(cuda_ver)
    
    def test_torch_cuda_detection(self):
        """Test detection when torch is installed with CUDA (integration test)."""
        # This is an integration test - only runs if torch is actually installed
        try:
            import torch
            python_ver, torch_ver, cuda_ver = find_wheel.detect_environment()
            
            # Verify we get valid results
            self.assertIsInstance(python_ver, tuple)
            self.assertEqual(len(python_ver), 2)
            
            self.assertIsInstance(torch_ver, tuple)
            self.assertEqual(len(torch_ver), 3)
            
            # CUDA version may be None (CPU build) or a tuple
            if cuda_ver is not None:
                self.assertIsInstance(cuda_ver, tuple)
                self.assertEqual(len(cuda_ver), 2)
        except ImportError:
            self.skipTest("torch not installed")
    
    def test_torch_cpu_detection(self):
        """Test that CPU-only torch results in None cuda_version."""
        # This is tested as part of torch_cuda_detection - if torch
        # is installed and cuda_ver is None, it's a CPU build
        # Skipping separate test to avoid complex mocking
        pass


class TestPipCommandFormatting(unittest.TestCase):
    """Test pip command formatting."""
    
    def test_format_pip_command(self):
        """Test pip command formatting with various configs."""
        base_url = 'https://by-ae.github.io/nvdiffrec_render-wheels/'
        
        cmd = find_wheel.format_pip_command('cu128-torch291', base_url)
        expected = 'pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu128-torch291/'
        self.assertEqual(cmd, expected)
        
        cmd = find_wheel.format_pip_command('cu124-torch251', base_url)
        expected = 'pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu124-torch251/'
        self.assertEqual(cmd, expected)
    
    def test_format_pip_command_custom_url(self):
        """Test pip command with custom base URL."""
        base_url = 'https://example.com/wheels/'
        
        cmd = find_wheel.format_pip_command('cu128-torch291', base_url)
        expected = 'pip install nvdiffrec_render --find-links https://example.com/wheels/cu128-torch291/'
        self.assertEqual(cmd, expected)

    def test_format_pip_command_adds_a_slash(self):
        """A base URL without a trailing slash still separates the config name."""
        cmd = find_wheel.format_pip_command(
            'cu128-torch291',
            'https://example.com/wheels',
        )
        expected = 'pip install nvdiffrec_render --find-links https://example.com/wheels/cu128-torch291/'
        self.assertEqual(cmd, expected)


class TestPythonRangeParsing(unittest.TestCase):
    """Test Python version range parsing."""
    
    def test_parse_python_range_standard(self):
        """Test parsing standard Python range strings."""
        result = find_wheel.parse_python_range('3.10-3.13')
        self.assertEqual(result, ((3, 10), (3, 13)))
        
        result = find_wheel.parse_python_range('3.10-3.12')
        self.assertEqual(result, ((3, 10), (3, 12)))
    
    def test_parse_python_range_invalid(self):
        """Test that invalid ranges return None."""
        self.assertIsNone(find_wheel.parse_python_range('3.10'))
        self.assertIsNone(find_wheel.parse_python_range('invalid'))
        self.assertIsNone(find_wheel.parse_python_range('3.10-3'))


class TestParseConfigsFromIndexHTML(unittest.TestCase):
    """Test parsing configs from index.html."""
    
    def test_parse_real_index_html(self):
        """Test parsing against the real wheels/index.html."""
        index_path = Path(__file__).parent / 'wheels' / 'index.html'
        
        if not index_path.exists():
            self.skipTest("wheels/index.html not found")
        
        with open(index_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        configs = find_wheel.parse_configs_from_index_html(html)
        
        # Published configurations only. cu130-torch291 has no wheels.
        self.assertEqual(
            {config['name'] for config in configs},
            {
                'cu124-torch251',
                'cu126-torch260',
                'cu126-torch280',
                'cu128-torch280',
                'cu128-torch291',
            },
        )
        
        # Check structure
        for config in configs:
            self.assertIn('name', config)
            self.assertIn('cuda', config)
            self.assertIn('torch', config)
            self.assertIsInstance(config['cuda'], tuple)
            self.assertIsInstance(config['torch'], tuple)
        
        # Check cu124-torch251 has Python range 3.10-3.12
        cu124 = next((c for c in configs if c['name'] == 'cu124-torch251'), None)
        if cu124:
            self.assertIn('python_range', cu124)
            self.assertEqual(cu124['python_range'], ((3, 10), (3, 12)))
        
        # Check cu128-torch291 has Python range 3.10-3.13
        cu128 = next((c for c in configs if c['name'] == 'cu128-torch291'), None)
        if cu128:
            self.assertIn('python_range', cu128)
            self.assertEqual(cu128['python_range'], ((3, 10), (3, 13)))
            self.assertEqual(cu128['platforms'], {'linux_x86_64', 'win_amd64'})
    
    def test_parse_mock_index_html(self):
        """Test parsing with mock HTML."""
        html = """
        <html>
        <body>
        <table>
            <tr><th>CUDA</th><th>PyTorch</th><th>Python</th><th>Link</th></tr>
            <tr><td>12.8</td><td>2.9.1</td><td>3.10-3.13</td><td><a href="cu128-torch291/">cu128-torch291</a></td></tr>
            <tr><td>12.4</td><td>2.5.1</td><td>3.10-3.12</td><td><a href="cu124-torch251/">cu124-torch251</a></td></tr>
        </table>
        </body>
        </html>
        """
        
        configs = find_wheel.parse_configs_from_index_html(html)
        
        self.assertEqual(len(configs), 2)
        
        # Check cu128-torch291
        cu128 = next((c for c in configs if c['name'] == 'cu128-torch291'), None)
        self.assertIsNotNone(cu128)
        self.assertEqual(cu128['cuda'], (12, 8))
        self.assertEqual(cu128['torch'], (2, 9, 1))
        self.assertEqual(cu128['python_range'], ((3, 10), (3, 13)))
        
        # Check cu124-torch251
        cu124 = next((c for c in configs if c['name'] == 'cu124-torch251'), None)
        self.assertIsNotNone(cu124)
        self.assertEqual(cu124['cuda'], (12, 4))
        self.assertEqual(cu124['torch'], (2, 5, 1))
        self.assertEqual(cu124['python_range'], ((3, 10), (3, 12)))


class TestCheckPythonVersionSupported(unittest.TestCase):
    """Test Python version support checking."""
    
    def test_python_version_in_range(self):
        """Test Python version within supported range."""
        config = {
            'name': 'cu128-torch291',
            'cuda': (12, 8),
            'torch': (2, 9, 1),
            'python_range': ((3, 10), (3, 13))
        }
        
        self.assertTrue(find_wheel.check_python_version_supported((3, 10), config))
        self.assertTrue(find_wheel.check_python_version_supported((3, 11), config))
        self.assertTrue(find_wheel.check_python_version_supported((3, 12), config))
        self.assertTrue(find_wheel.check_python_version_supported((3, 13), config))
    
    def test_python_version_out_of_range(self):
        """Test Python version outside supported range."""
        config = {
            'name': 'cu124-torch251',
            'cuda': (12, 4),
            'torch': (2, 5, 1),
            'python_range': ((3, 10), (3, 12))
        }
        
        self.assertFalse(find_wheel.check_python_version_supported((3, 9), config))
        self.assertFalse(find_wheel.check_python_version_supported((3, 13), config))
        self.assertFalse(find_wheel.check_python_version_supported((3, 14), config))
    
    def test_python_version_no_range(self):
        """Test that configs without python_range always pass."""
        config = {
            'name': 'cu128-torch291',
            'cuda': (12, 8),
            'torch': (2, 9, 1)
        }
        
        # Should pass for any Python version when no range is specified
        self.assertTrue(find_wheel.check_python_version_supported((3, 9), config))
        self.assertTrue(find_wheel.check_python_version_supported((3, 10), config))
        self.assertTrue(find_wheel.check_python_version_supported((3, 13), config))


class TestFetchRemoteIndex(unittest.TestCase):
    """Test remote index fetching."""
    
    def test_fetch_remote_index_success(self):
        """Test successful fetch with mocked urlopen."""
        mock_html = b'<html><body>test</body></html>'
        
        with patch('find_wheel.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = mock_html
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_urlopen.return_value = mock_response
            
            result = find_wheel.fetch_remote_index('http://example.com/index.html')
            
            self.assertEqual(result, '<html><body>test</body></html>')
            mock_urlopen.assert_called_once()
    
    def test_fetch_remote_index_failure(self):
        """Test fetch failure with mocked urlopen."""
        with patch('find_wheel.urlopen') as mock_urlopen:
            from urllib.error import URLError
            mock_urlopen.side_effect = URLError('Connection failed')
            
            result = find_wheel.fetch_remote_index('http://example.com/index.html')
            
            self.assertIsNone(result)


class TestScriptWheelsDirectory(unittest.TestCase):
    def test_stdin_has_no_local_wheels_directory(self):
        self.assertIsNone(find_wheel.wheels_dir_for_script(None))
        self.assertIsNone(find_wheel.wheels_dir_for_script(''))
        self.assertIsNone(find_wheel.wheels_dir_for_script('<stdin>'))
        self.assertIsNone(find_wheel.wheels_dir_for_script('-'))
        script = Path(__file__).resolve()
        self.assertEqual(
            find_wheel.wheels_dir_for_script(str(script)),
            script.parent / 'wheels',
        )


class TestHostPlatform(unittest.TestCase):
    def test_published_tags(self):
        self.assertEqual(find_wheel.host_wheel_platform("Linux", "x86_64"), "linux_x86_64")
        self.assertEqual(find_wheel.host_wheel_platform("Linux", "amd64"), "linux_x86_64")
        self.assertEqual(find_wheel.host_wheel_platform("Windows", "AMD64"), "win_amd64")
        self.assertEqual(find_wheel.host_wheel_platform("Windows", "x86_64"), "win_amd64")

    def test_unpublished_tags(self):
        self.assertIsNone(find_wheel.host_wheel_platform("Darwin", "arm64"))
        self.assertIsNone(find_wheel.host_wheel_platform("Darwin", "x86_64"))
        self.assertIsNone(find_wheel.host_wheel_platform("Linux", "aarch64"))
        self.assertIsNone(find_wheel.host_wheel_platform("Windows", "ARM64"))


class TestInstallCommand(unittest.TestCase):
    """The script must not print a pip command for a different build."""

    def run_finder(self, *args):
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "find_wheel.py"), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_exact_match_prints_one_install_command(self):
        result = self.run_finder("--torch", "2.9.1", "--cuda", "12.8", "--python", "3.11")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "pip install nvdiffrec_render --find-links "
            "https://by-ae.github.io/nvdiffrec_render-wheels/cu128-torch291/",
            result.stdout,
        )
        self.assertEqual(result.stdout.count("pip install"), 1)

    def test_cuda_13_does_not_offer_the_cu128_wheel(self):
        result = self.run_finder("--torch", "2.9.1", "--cuda", "13.0", "--python", "3.11")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("pip install", result.stdout)
        self.assertNotIn("--find-links", result.stdout)
        self.assertIn("cu128-torch291", result.stdout)
        self.assertIn("will not import", result.stdout)

    def test_different_torch_patch_does_not_offer_an_install_command(self):
        result = self.run_finder("--torch", "2.9.0", "--cuda", "12.8", "--python", "3.11")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("pip install", result.stdout)
        self.assertIn("will not import", result.stdout)

    def _run_main(self, argv, system, machine):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["find_wheel.py", *argv]), \
             patch("find_wheel.platform.system", return_value=system), \
             patch("find_wheel.platform.machine", return_value=machine), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                find_wheel.main()
        return raised.exception.code, stdout.getvalue(), stderr.getvalue()

    def test_windows_exact_match_still_prints_the_install_command(self):
        code, stdout, stderr = self._run_main(
            ["--torch", "2.9.1", "--cuda", "12.8", "--python", "3.11"],
            "Windows",
            "AMD64",
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("win_amd64", stdout)
        self.assertIn("cu128-torch291/", stdout)
        self.assertEqual(stdout.count("pip install"), 1)

    def test_unmatched_platform_does_not_print_an_install_command(self):
        code, stdout, stderr = self._run_main(
            ["--torch", "2.9.1", "--cuda", "12.8", "--python", "3.11"],
            "Darwin",
            "arm64",
        )
        self.assertNotEqual(code, 0)
        self.assertNotIn("pip install", stdout)
        self.assertIn("linux_x86_64", stdout)
        self.assertIn("win_amd64", stdout)

    def test_config_without_this_platform_tag_is_not_offered(self):
        html = """
        <tr><td>12.8</td><td>2.9.1</td><td>3.10-3.13</td><td><a href="cu128-torch291/">cu128-torch291</a></td></tr>
        <a href="https://example.com/cu128-torch291/nvdiffrec_render-0.0.1+cu128torch29-cp310-cp310-win_amd64.whl">w</a>
        """
        configs = find_wheel.parse_configs_from_index_html(html)
        self.assertEqual(configs[0]["platforms"], {"win_amd64"})
        message = find_wheel.platform_refusal("linux_x86_64", configs[0])
        self.assertIn("no linux_x86_64 wheel", message)
        self.assertIsNone(find_wheel.platform_refusal("win_amd64", configs[0]))

    def test_non_numeric_python_exits_without_a_traceback(self):
        result = self.run_finder("--torch", "2.9.1", "--cuda", "12.8", "--python", "abc")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Invalid Python version format", result.stderr)
        self.assertIn("Expected format: X.Y", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("Traceback", result.stdout)

    def test_python_patch_still_matches(self):
        result = self.run_finder("--torch", "2.9.1", "--cuda", "12.8", "--python", "3.11.4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cu128-torch291/", result.stdout)

    def test_piped_script_ignores_wheels_in_the_working_directory(self):
        script = (Path(__file__).resolve().parent / "find_wheel.py").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            wheels = Path(tmp) / "wheels"
            wheels.mkdir()
            (wheels / "index.html").write_text(
                '<tr><td>11.1</td><td>1.2.3</td><td>3.10-3.13</td>'
                '<td><a href="cu111-torch123/">cu111-torch123</a></td></tr>\n',
                encoding="utf-8",
            )
            piped = subprocess.run(
                [sys.executable, "-", "--torch", "1.2.3", "--cuda", "11.1", "--python", "3.11"],
                input=script,
                text=True,
                capture_output=True,
                cwd=tmp,
                check=False,
            )
            combined = piped.stdout + piped.stderr
            self.assertNotIn("cu111-torch123", combined)
            self.assertNotIn("Traceback", combined)

            local = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "find_wheel.py"),
                    "--wheels-dir",
                    str(wheels),
                    "--torch",
                    "1.2.3",
                    "--cuda",
                    "11.1",
                    "--python",
                    "3.11",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(local.returncode, 0, local.stderr)
        self.assertIn("cu111-torch123/", local.stdout)
        self.assertEqual(local.stdout.count("pip install"), 1)


if __name__ == '__main__':
    unittest.main()
