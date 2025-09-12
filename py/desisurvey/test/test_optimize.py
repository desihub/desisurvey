import unittest

import numpy as np

from desisurvey.optimize import *


class TestUtils(unittest.TestCase):

    def test_wrap_unwrap(self):
        x = np.linspace(0., 350., 97)
        for dx in (-60, 0, 60):
            w = wrap(x, dx)
            assert np.all(w >= dx)
            assert np.all(w < dx + 360)
