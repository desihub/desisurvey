# DESI Survey Data

- `config.yaml`: Configuration YAML file, used by `config.Configuration()`.
- `iers_frozen.ecsv`: Tabulated UT1-UTC and polar motion in the IERS format required by astropy time, coordinates.  Written by `utils.update_iers()` and read by `utils.freeze_iers()`.
- `tile-info.fits`: Design hour-angle windows for each tile, used by `afternoonplan`.
- `horizons_2020_week1_moon.csv`: Moon ephemerides for the first week of 2020 calculated using JPL Horizons and used for unit tests.
