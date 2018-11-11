import unittest

import numpy as np

import astropy.units as u

import desisurvey.utils
import desisurvey.config
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

    def test_overlap(self):
        tiles = Tiles()
        # Check that overlap attributes are cached.
        tile_over = tiles.tile_over
        self.assertEqual(id(tile_over), id(tiles.tile_over))
        overlapping = tiles.overlapping
        self.assertEqual(id(overlapping), id(tiles.overlapping))
        # Sanity checks on overlap results.
        # Assume that the second DARK pass depends on the first DARK pass.
        DARK1, DARK2 = tiles.program_passes['DARK'][:2]
        print(DARK1, DARK2)
        IN1 = (tiles.passnum == DARK1)
        IN2 = (tiles.passnum == DARK2)
        self.assertEqual(len(tile_over[DARK1]), tiles.ntiles)
        self.assertFalse(np.any(tile_over[DARK1]))
        self.assertTrue(np.all(tile_over[DARK2][IN1]))
        self.assertFalse(np.any(tile_over[DARK2][IN2]))

        self.assertTrue(DARK1 not in overlapping)
        self.assertEqual(overlapping[DARK2].shape, (tiles.pass_ntiles[DARK2], tiles.pass_ntiles[DARK1]))
        # Pick a tile in the second DARK pass.
        N = 500
        IDX2 = np.where(IN2)[0][N]
        # Find covering tiles in the first DARK pass.
        IDX1 = np.where(tile_over[DARK2])[0][overlapping[DARK2][N]]
        # Calculate separations.
        sep = desisurvey.utils.separation_matrix(
            [tiles.tileRA[IDX2]], [tiles.tileDEC[IDX2]],
            tiles.tileRA[IDX1], tiles.tileDEC[IDX1])
        self.assertEqual(sep.shape, (1, len(IDX1)))
        config = desisurvey.config.Configuration()
        self.assertTrue(np.max(sep) <= 2 * config.tile_radius().to(u.deg).value)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
