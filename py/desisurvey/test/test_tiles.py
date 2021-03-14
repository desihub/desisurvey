import unittest

import numpy as np

import astropy.units as u

import desisurvey.utils
import desisurvey.config
from desisurvey.test.base import Tester
from desisurvey.tiles import Tiles, get_tiles


class TestTiles(Tester):

    def test_tiles(self):
        tiles = Tiles()
        # Verify tile indexing round trip.
        assert np.array_equal(tiles.index(tiles.tileID), np.arange(tiles.ntiles))
        # Verify pass indexing round trip.
        for program in tiles.programs:
            assert tiles.programs[tiles.program_index[program]] == program
        # Check reasonable dust and airmass values.
        assert np.all(tiles.dust_factor > 1)
        assert np.all(tiles.airmass(np.zeros(tiles.ntiles)) >= 1)

    def test_get(self):
        tiles1 = get_tiles()
        tiles2 = get_tiles()
        assert id(tiles1) == id(tiles2)

    def test_overlap(self):
        tiles = Tiles()
        overlapping = tiles.overlapping
        self.assertEqual(id(overlapping), id(tiles.overlapping))
        # Sanity checks on overlap results.
        # Assume that the DARK program has some overlapping requirements.
        darkidx = np.flatnonzero(tiles.program_mask['DARK'])
        brightidx = np.flatnonzero(tiles.program_mask['BRIGHT'])
        overlapping = tiles.overlapping
        self.assertTrue(len(overlapping[darkidx[0]]) > 0)
        # check that bright and dark don't overlap
        self.assertFalse(np.any(np.isin(brightidx, overlapping[darkidx[0]])))
        self.assertFalse(np.any(np.isin(darkidx, overlapping[brightidx[0]])))
        config = desisurvey.config.Configuration()
        for N in range(0, 25, 5):
            IDX2 = darkidx[N]
            # Find overlapping tiles
            IDX1 = overlapping[IDX2]
            # Calculate separations.
            sep = desisurvey.utils.separation_matrix(
                [tiles.tileRA[IDX2]], [tiles.tileDEC[IDX2]],
                tiles.tileRA[IDX1], tiles.tileDEC[IDX1])
            self.assertEqual(sep.shape, (1, len(IDX1)))
            self.assertTrue(np.max(sep) <= 2 * config.tile_radius().to(u.deg).value)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
