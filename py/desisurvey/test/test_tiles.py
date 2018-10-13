import unittest

import numpy as np

import astropy.units as u

from desisurvey.tiles import Tiles, get_tiles


class TestTiles(unittest.TestCase):

    def setUp(self):
        pass

    def test_tiles(self):
        tiles = Tiles()
        # Verify tile indexing round trip.
        assert np.array_equal(tiles.index(tiles.tileID), np.arange(tiles.ntiles))
        # Verify pass indexing round trip.
        for passnum in tiles.passes:
            assert tiles.passes[tiles.pass_index[passnum]] == passnum
        # Check reasonable dust and airmass values.
        assert np.all(tiles.dust_factor > 1)
        assert np.all(tiles.airmass(np.zeros(tiles.ntiles)) >= 1)

    def test_get(self):
        tiles1 = get_tiles()
        tiles2 = get_tiles()
        assert id(tiles1) == id(tiles2)
