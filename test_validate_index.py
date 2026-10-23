#!/usr/bin/env python3
"""
Unit tests for validate_index.py.
Tests the index HTML validation logic without requiring actual wheel files.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path to import validate_index
sys.path.insert(0, str(Path(__file__).parent))
import validate_index


class TestWheelFilenameValidation(unittest.TestCase):
    """Test wheel filename validation."""
    
    def test_valid_wheel_names(self):
        """Test that valid wheel names pass validation."""
        valid_names = [
            "nvdiffrec_render-0.0.1+cu128torch29-cp310-cp310-linux_x86_64.whl",
            "nvdiffrec_render-0.0.1+cu124torch25-cp312+cu124torch25-cp312-win_amd64.whl",
            "nvdiffrec_render-0.0.1+cu126torch26-cp313-cp313-linux_x86_64.whl",
        ]
        
        for name in valid_names:
            with self.subTest(name=name):
                valid, error = validate_index.validate_wheel_filename(name)
                self.assertTrue(valid, f"Expected {name} to be valid, but got: {error}")
    
    def test_invalid_wheel_names(self):
        """Test that invalid wheel names fail validation."""
        invalid_names = [
            "notawheelfile.txt",
            "wrong_package-0.0.1-cp310-cp310-linux_x86_64.whl",
            "nvdiffrec_render-0.0.1.tar.gz",
            "nvdiffrec_render-0.0.1",
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                valid, error = validate_index.validate_wheel_filename(name)
                self.assertFalse(valid, f"Expected {name} to be invalid but passed")


class TestIndexHTMLParser(unittest.TestCase):
    """Test HTML parsing functionality."""
    
    def test_parse_wheel_links(self):
        """Test parsing wheel links from HTML."""
        html = """
        <html>
        <body>
        <a href="wheel1.whl">wheel1</a>
        <a href="https://example.com/wheel2.whl">wheel2</a>
        </body>
        </html>
        """
        
        parser = validate_index.IndexHTMLParser()
        parser.feed(html)
        
        self.assertEqual(len(parser.links), 2)
        self.assertIn("wheel1.whl", parser.links)
        self.assertIn("https://example.com/wheel2.whl", parser.links)
    
    def test_parse_subdirectory_links(self):
        """Test parsing subdirectory links from HTML."""
        html = """
        <html>
        <body>
        <a href="cu128-torch291/">cu128-torch291</a>
        <a href="cu126-torch280/">cu126-torch280</a>
        <a href="https://github.com/external">external</a>
        </body>
        </html>
        """
        
        parser = validate_index.IndexHTMLParser()
        parser.feed(html)
        
        self.assertEqual(len(parser.subdirs), 3)
        self.assertIn("cu128-torch291", parser.subdirs)
        self.assertIn("cu126-torch280", parser.subdirs)


class TestSubindexValidation(unittest.TestCase):
    """Test subindex validation with temporary files."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
    
    def test_valid_subindex(self):
        """Test validation of a valid subindex."""
        index_path = Path(self.test_dir) / "index.html"
        index_path.write_text("""
        <!DOCTYPE html>
        <html>
        <body>
        <h1>Wheels</h1>
        <a href="nvdiffrec_render-0.0.1+cu128torch29-cp310-cp310-linux_x86_64.whl">wheel</a>
        </body>
        </html>
        """)
        
        errors, warnings = validate_index.validate_subindex(index_path)
        
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
    
    def test_missing_index(self):
        """Test validation of missing index file."""
        index_path = Path(self.test_dir) / "nonexistent.html"
        
        errors, warnings = validate_index.validate_subindex(index_path)
        
        self.assertEqual(len(errors), 1)
        self.assertIn("not found", errors[0])
    
    def test_malformed_html(self):
        """Test validation of malformed HTML."""
        index_path = Path(self.test_dir) / "index.html"
        index_path.write_text("<body>incomplete")
        
        errors, warnings = validate_index.validate_subindex(index_path)
        
        # Should have error about missing HTML tags
        self.assertGreater(len(errors), 0)


class TestTopIndexValidation(unittest.TestCase):
    """Test top-level index validation."""
    
    def setUp(self):
        """Create temporary directory structure."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
        
        # Create some subdirectories with index files
        for subdir in ['cu128-torch291', 'cu126-torch280']:
            subdir_path = Path(self.test_dir) / subdir
            subdir_path.mkdir()
            (subdir_path / 'index.html').write_text('<html><body>test</body></html>')
    
    def test_valid_top_index(self):
        """Test validation when all subdirs are linked."""
        index_path = Path(self.test_dir) / "index.html"
        index_path.write_text("""
        <html>
        <body>
        <a href="cu128-torch291/">cu128-torch291</a>
        <a href="cu126-torch280/">cu126-torch280</a>
        </body>
        </html>
        """)
        
        errors, warnings = validate_index.validate_top_index(index_path, Path(self.test_dir))
        
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
    
    def test_missing_subdir_link(self):
        """Test validation when a subdir is not linked."""
        index_path = Path(self.test_dir) / "index.html"
        index_path.write_text("""
        <html>
        <body>
        <a href="cu128-torch291/">cu128-torch291</a>
        </body>
        </html>
        """)
        
        errors, warnings = validate_index.validate_top_index(index_path, Path(self.test_dir))
        
        self.assertEqual(len(errors), 1)
        self.assertIn("cu126-torch280", errors[0])
    
    def test_external_links_ignored(self):
        """Test that external links don't cause false positives."""
        index_path = Path(self.test_dir) / "index.html"
        index_path.write_text("""
        <html>
        <body>
        <a href="cu128-torch291/">cu128-torch291</a>
        <a href="cu126-torch280/">cu126-torch280</a>
        <a href="https://github.com/external">GitHub</a>
        </body>
        </html>
        """)
        
        errors, warnings = validate_index.validate_top_index(index_path, Path(self.test_dir))
        
        # External link should not cause errors
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    unittest.main()
