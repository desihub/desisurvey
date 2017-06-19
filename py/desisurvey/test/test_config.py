from __future__ import print_function, division

import unittest
import os
import tempfile
import shutil

import numpy as np

import astropy.units

from ..config import *


class TestConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory.
        cls.tmpdir = tempfile.mkdtemp()
        # Write a simple valid configuration to this directory.
        cls.simple = os.path.join(cls.tmpdir, 'simple.yaml')
        with open(cls.simple, 'w') as f:
            f.write('const:\n')
            f.write('  gravity: 9.8 m/s\n')
            f.write('  pi: 3.141\n')
            f.write('output_path: .\n')

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test.
        shutil.rmtree(cls.tmpdir)
        # Reset configuration for other unit test classes.
        Configuration.reset()

    def test_default(self):
        """Default config file is valid"""
        Configuration.reset()
        c = Configuration()

    def test_repeat_default(self):
        """Repeated calls to default init return same object"""
        Configuration.reset()
        c1 = Configuration()
        c2 = Configuration()
        self.assertEqual(id(c1), id(c2))

    def test_repeat_non_default(self):
        """Repeated calls to non-default init return same object"""
        Configuration.reset()
        c1 = Configuration(self.simple)
        c2 = Configuration(self.simple)
        self.assertEqual(id(c1), id(c2))

    def test_valid(self):
        """Simple config file really is valid"""
        Configuration.reset()
        c = Configuration(self.simple)

    def test_terminal_value(self):
        """Terminal node values accessed via __call__"""
        Configuration.reset()
        c = Configuration(self.simple)
        self.assertEqual(c.const.gravity(), astropy.units.Quantity(9.8, 'm/s'))
        self.assertEqual(c.const.pi(), 3.141)
        self.assertEqual(c.output_path(), '.')

    def test_non_terminal_value(self):
        """Non-terminal nodes do not have an associated value"""
        Configuration.reset()
        c = Configuration(self.simple)
        with self.assertRaises(RuntimeError):
            c.const()

    def test_node_path(self):
        """Config nodes have associated paths"""
        Configuration.reset()
        c = Configuration(self.simple)
        self.assertEqual(c.const.gravity.path, 'const.gravity')
        self.assertEqual(c.output_path.path, 'output_path')
        Configuration.reset()
        c = Configuration()
        self.assertEqual(c.programs.DARK.path, 'programs.DARK')

    def test_get_path_rel(self):
        """A relative path has output_path prepended"""
        Configuration.reset()
        c = Configuration(self.simple)
        self.assertEqual(c.get_path('blah'), os.path.join('.', 'blah'))

    def test_get_path_abs(self):
        """An absolute path does not have output_path prepended"""
        Configuration.reset()
        c = Configuration(self.simple)
        self.assertEqual(c.get_path('/blah'), '/blah')

    def test_set_output_path(self):
        """output_path can be changed"""
        Configuration.reset()
        c = Configuration(self.simple)
        os.environ['_OUTPUT_PATH_'] = self.tmpdir
        c.set_output_path('{_OUTPUT_PATH_}')
        self.assertEqual(c.output_path(), '{_OUTPUT_PATH_}')
        self.assertEqual(c.get_path('blah'), os.path.join(self.tmpdir, 'blah'))
        with self.assertRaises(ValueError):
            c.set_output_path('{_OUTPUT_PATH_}/_non_existent_')
        del os.environ['_OUTPUT_PATH_']

    def test_bad_abs_path(self):
        """Cannot read config from a non-existent absolute path"""
        Configuration.reset()
        name = os.path.join(self.tmpdir, '_non_existent_.yaml')
        with self.assertRaises(IOError):
            c = Configuration(name)

    def test_bad_rel_path(self):
        """Cannot read config from a non-existent relative path"""
        Configuration.reset()
        with self.assertRaises(IOError):
            c = Configuration('_non_existent_.yaml')

    def test_change_name(self):
        """Cannot read config from differerent files in same session."""
        Configuration.reset()
        c = Configuration('config.yaml')
        with self.assertRaises(RuntimeError):
            c = Configuration(self.simple)
        Configuration.reset()
        c = Configuration(self.simple)
        with self.assertRaises(RuntimeError):
            c = Configuration('config.yaml')

    def test_output_path(self):
        """output_path must exist"""
        bad = os.path.join(self.tmpdir, 'bad.yaml')
        with open(bad, 'w') as f:
            f.write('output_path: _non_existent_\n')
        Configuration.reset()
        c = Configuration(bad)
        with self.assertRaises(ValueError):
            c.get_path('blah')

    def test_dict_key_type(self):
        """Dictionary keys must be valid python identifiers"""
        bad = os.path.join(self.tmpdir, 'bad.yaml')
        with open(bad, 'w') as f:
            f.write('[1, 2]: 123')
            f.write('output_path: .\n')
        Configuration.reset()
        with self.assertRaises(RuntimeError):
            c = Configuration(bad)

    def test_dict_key_name(self):
        """Dictionary keys must be valid python identifiers"""
        bad = os.path.join(self.tmpdir, 'bad.yaml')
        with open(bad, 'w') as f:
            f.write('invalid-key: 123')
            f.write('output_path: .\n')
        Configuration.reset()
        with self.assertRaises(RuntimeError):
            c = Configuration(bad)

    def test_no_seq(self):
        """YAML sequences are not allowed"""
        bad = os.path.join(self.tmpdir, 'bad.yaml')
        with open(bad, 'w') as f:
            f.write('key: [1, 2]')
            f.write('output_path: .\n')
        Configuration.reset()
        with self.assertRaises(RuntimeError):
            c = Configuration(bad)

    def test_keys(self):
        """Non-terminal node has keys property"""
        Configuration.reset()
        c = Configuration()
        k = c.avoid_bodies.keys
        self.assertTrue('moon' in k)
        self.assertTrue('jupiter' in k)

    def test_terminal_keys(self):
        """Terminal nodes have no keys"""
        Configuration.reset()
        c = Configuration()
        with self.assertRaises(RuntimeError):
            k = c.avoid_bodies.moon.keys


if __name__ == '__main__':
    unittest.main()
