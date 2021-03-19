import numpy as np
import desisurvey.tiles


def donefrac_nnight(numnight, configfn=None):
    """Get donefrac corresponding to number of tiles observed
    in different conditions.

    condnexp from desisurvey.scripts.collect_etc.number_in_conditions
    """
    import desisurvey.config
    config = desisurvey.config.Configuration(configfn)
    tiles = desisurvey.tiles.get_tiles()
    out = np.zeros(tiles.ntiles, dtype=[
        ('TILEID', 'i4'), ('NNIGHT', 'f4'), ('NNIGHT_NEEDED', 'f4')
    ])
    if ((len(np.unique(numnight['TILEID'])) != len(numnight)) or
            (len(np.unique(tiles.tileID)) != tiles.ntiles)):
        raise ValueError('Must be at least one tile per record!')
    if np.any(numnight['TILEID'] < 0):
        raise ValueError('tileID must be >= 0 for real exposures!')
    _, mn, mt = np.intersect1d(numnight['TILEID'], tiles.tileID,
                               return_indices=True)
    out['TILEID'] = tiles.tileID
    out['NNIGHT'][mt] = numnight['NNIGHT'][mn]
    for i, program in enumerate(tiles.tileprogram):
        needed = getattr(config.programs, program, 0)
        if not isinstance(needed, int):
            needed = getattr(needed, 'nnight_needed', None)
            needed = 0 if needed is None else needed()
            out['NNIGHT_NEEDED'][i] = needed
    return out
