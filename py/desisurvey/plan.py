"""Plan future DESI observations.
"""
from __future__ import print_function, division

import numpy as np

import astropy.io.fits as fits
import astropy.units as u
import astropy.table
import astropy.time

import specsim.atmosphere

import desimodel.io

import desiutil.log

import desisurvey.ephemerides
import desisurvey.exposurecalc
import desisurvey.config
import desisurvey.utils


def initialize(ephem, start_date=None, stop_date=None, step_size=5.*u.min,
               healpix_nside=16, output_name='planner.fits'):
    """Calculate exposure-time factors over a grid of times and pointings.

    Takes about 6 minutes to run and writes a 1.3Gb output file with the
    default parameters.

    Requires that healpy is installed.

    Parameters
    ----------
    ephem : `desisurvey.ephem.Ephemerides`
        Tabulated ephemerides data to use for planning.
    start_date : date or None
        Survey planning starts on the evening of this date. Must be convertible
        to a date using :func:`desisurvey.utils.get_date`.  Use the first night
        of the ephemerides when None.
    stop_date : date or None
        Survey planning stops on the morning of this date. Must be convertible
        to a date using :func:`desisurvey.utils.get_date`.  Use the first night
        of the ephemerides when None.
    step_size : astropy.units.Quantity
        Exposure-time factors are tabulated at this interval during each night.
    healpix_nside : int
        Healpix NSIDE parameter to use for binning the sky. Must be a power of
        two.  Values larger than 16 will lead to holes in the footprint with
        the current implementation.
    output_name : str
        Name of the FITS output file where results are saved. A relative path
        refers to the :meth:`configuration output path
        <desisurvey.config.Configuration.get_path>`.
    """
    import healpy

    log = desiutil.log.get_logger()

    config = desisurvey.config.Configuration()
    output_name = config.get_path(output_name)

    start_date = desisurvey.utils.get_date(start_date or ephem.start)
    stop_date = desisurvey.utils.get_date(stop_date or ephem.stop)
    if start_date >= stop_date:
        raise ValueError('Expected start_date < stop_date.')
    mjd = ephem._table['noon']
    sel = ((mjd >= desisurvey.utils.local_noon_on_date(start_date).mjd) &
           (mjd < desisurvey.utils.local_noon_on_date(stop_date).mjd))
    t = ephem._table[sel]
    num_nights = len(t)

    # Build a grid of elapsed time relative to local midnight during each night.
    midnight = t['noon'] + 0.5
    t_edges = desisurvey.ephemerides.get_grid(step_size)
    t_centers = 0.5 * (t_edges[1:] + t_edges[:-1])
    num_points = len(t_centers)

    # Create an empty HDU0 with header info.
    header = fits.Header()
    header['START'] = str(start_date)
    header['STOP'] = str(stop_date)
    header['NSIDE'] = healpix_nside
    header['NPOINTS'] = num_points
    header['STEP'] = step_size.to(u.min).value
    hdus = fits.HDUList()
    hdus.append(fits.ImageHDU(header=header))

    # Save time grid.
    hdus.append(fits.ImageHDU(name='GRID', data=t_edges))

    # Load the list of tiles to observe.
    tiles = astropy.table.Table(
        desimodel.io.load_tiles(onlydesi=True, extra=False))

    # Build the footprint as a healpix map of the requested size.
    # The footprint includes any pixel containing at least one tile center.
    npix = healpy.nside2npix(healpix_nside)
    footprint = np.zeros(npix, bool)
    pixels = healpy.ang2pix(
            healpix_nside, np.radians(90 - tiles['DEC'].data),
            np.radians(tiles['RA'].data))
    footprint[np.unique(pixels)] = True
    footprint_pixels = np.where(footprint)[0]
    num_footprint = len(footprint_pixels)
    log.info('Footprint contains {0} pixels.'.format(num_footprint))

    # Sort pixels in order of increasing phi + 60deg so that the north and south
    # galactic caps are contiguous in the arrays we create below.
    pix_theta, pix_phi = healpy.pix2ang(healpix_nside, footprint_pixels)
    pix_dphi = np.fmod(pix_phi + np.pi / 3, 2 * np.pi)
    sort_order = np.argsort(pix_dphi)
    footprint_pixels = footprint_pixels[sort_order]

    # Average E(B-V) for all tiles falling into a pixel.
    tiles_per_pixel = np.bincount(pixels, minlength=npix)
    EBV = np.bincount(pixels, weights=tiles['EBV_MED'], minlength=npix)
    EBV[footprint] /= tiles_per_pixel[footprint]

    # Calculate dust extinction exposure-time factor.
    Ag = 3.303 * EBV # Use g-band
    f_EBV = 10.0 ** (-2.0 * Ag / 2.5)

    # Save HDU with the footprint and static dust exposure map.
    table = astropy.table.Table()
    table['pixel'] = footprint_pixels
    table['dust'] = f_EBV[footprint_pixels]
    hdu = fits.table_to_hdu(table)
    hdu.name = 'STATIC'
    hdus.append(hdu)

    # Prepare a table of calendar data.
    calendar = astropy.table.Table()
    calendar['midnight'] = midnight
    calendar['monsoon'] = np.zeros(num_nights, bool)
    calendar['fullmoon'] = np.zeros(num_nights, bool)

    # Prepare a table of ephemeris data.
    etable = astropy.table.Table()
    etable['program'] = np.zeros(num_nights * num_points, dtype=np.int16)
    etable['moon_frac'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['moon_ra'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['moon_dec'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['zenith_ra'] = np.zeros(num_nights * num_points, dtype=np.float32)
    etable['zenith_dec'] = np.zeros(num_nights * num_points, dtype=np.float32)

    # Build sky coordinates for each pixel in the footprint.
    pix_theta, pix_phi = healpy.pix2ang(healpix_nside, footprint_pixels)
    pix_ra, pix_dec = np.degrees(pix_phi), 90 - np.degrees(pix_theta)
    pix_sky = astropy.coordinates.ICRS(pix_ra * u.deg, pix_dec * u.deg)

    # Initialize exposure factor calculations.
    alt, az = np.full(num_points, 90.) * u.deg, np.zeros(num_points) * u.deg
    fexp = np.zeros((num_nights * num_points, num_footprint), dtype=np.float32)
    vband_extinction = 0.15154
    one = np.ones((num_points, num_footprint))

    # Loop over nights.
    for i in range(num_nights):
        night = ephem.get_night(midnight[i])
        date = desisurvey.utils.get_date(midnight[i])
        if date.day == 1:
            log.info('Starting {0} (completed {1}/{2} nights)'
                     .format(date.strftime('%b %Y'), i, num_nights))
        # Initialize the slice of the fexp[] time index for this night.
        sl = slice(i * num_points, (i + 1) * num_points)
        # Skip monsoon and full moon.
        calendar[i]['monsoon'] = desisurvey.utils.is_monsoon(midnight[i])
        calendar[i]['fullmoon'] = ephem.is_full_moon(midnight[i])
        if calendar[i]['monsoon'] or calendar[i]['fullmoon']:
            continue
        # Calculate the program during this night.
        mjd = midnight[i] + t_centers
        dark, gray, bright = ephem.get_program(mjd)
        etable['program'][sl][dark] = 1
        etable['program'][sl][gray] = 2
        etable['program'][sl][bright] = 3
        # Zero the exposure factor whenever we are not oberving.
        fexp[sl] = (dark | gray | bright)[:, np.newaxis]
        # Transform the local zenith to (ra,dec).
        times = astropy.time.Time(mjd, format='mjd')
        zenith = desisurvey.utils.get_observer(
            times, alt=alt, az=az).transform_to(astropy.coordinates.ICRS)
        etable['zenith_ra'][sl] = zenith.ra.to(u.deg).value
        etable['zenith_dec'][sl] = zenith.dec.to(u.deg).value
        # Calculate zenith angles to each pixel in the footprint.
        pix_sep = pix_sky.separation(zenith[:, np.newaxis])
        # Zero the exposure factor for pixels below the horizon.
        visible = pix_sep < 90 * u.deg
        fexp[sl][~visible] = 0.
        # Calculate the airmass exposure-time penalty
        # as 1/X**1.25 with X = 1/cos(pix_sep).
        fexp[sl][visible] *= np.cos(pix_sep[visible]) ** 1.25
        # Loop over objects we need to avoid.
        for name in config.avoid_bodies.keys:
            f_obj = desisurvey.ephemerides.get_object_interpolator(night, name)
            # Calculate this object's (dec,ra) path during the night.
            obj_dec, obj_ra = f_obj(times.mjd)
            sky_obj = astropy.coordinates.ICRS(
                ra=obj_ra[:, np.newaxis] * u.deg,
                dec=obj_dec[:, np.newaxis] * u.deg)
            # Calculate the separation angles to each pixel in the footprint.
            obj_sep = pix_sky.separation(sky_obj)
            if name == 'moon':
                etable['moon_ra'][sl] = obj_ra
                etable['moon_dec'][sl] = obj_dec
                # Calculate moon altitude during the night.
                moon_alt, _ = desisurvey.ephemerides.get_object_interpolator(
                    night, 'moon', altaz=True)(times.mjd)
                moon_zenith = (90 - moon_alt[:,np.newaxis]) * u.deg
                moon_up = moon_alt > 0
                assert np.all(moon_alt[gray] > 0)
                # Calculate the moon illuminated fraction during the night.
                moon_frac = ephem.get_moon_illuminated_fraction(times.mjd)
                etable['moon_frac'][sl] = moon_frac
                # Convert to temporal moon phase.
                moon_phase = np.arccos(2 * moon_frac[:,np.newaxis] - 1) / np.pi
                # Calculate scattered moon V-band brightness at each pixel.
                V = specsim.atmosphere.krisciunas_schaefer(
                    pix_sep, moon_zenith, obj_sep,
                    moon_phase, vband_extinction).value
                # Estimate the exposure time factor from V.
                X = np.dstack((one, np.exp(-V), 1/V, 1/V**2, 1/V**3))
                T = X.dot(desisurvey.exposurecalc._moonCoefficients)
                # No penalty when the moon is below the horizon.
                T[moon_alt < 0, :] = 1.
                fexp[sl] *= 1. / T
            else:
                # Lookup the avoidance size for this object.
                size = getattr(config.avoid_bodies, name)()
                # Penalize the exposure-time with a factor
                # 1 - exp(-0.5*(obj_sep/size)**2)
                penalty = 1. - np.exp(-0.5 * (obj_sep / size) ** 2)
                fexp[sl] *= penalty

    # Save calendar table.
    hdu = fits.table_to_hdu(calendar)
    hdu.name = 'CALENDAR'
    hdus.append(hdu)

    # Save ephemerides table.
    hdu = fits.table_to_hdu(etable)
    hdu.name = 'EPHEM'
    hdus.append(hdu)

    # Save dynamic exposure-time factors.
    hdus.append(fits.ImageHDU(name='DYNAMIC', data=fexp))

    # Finalize the output file.
    hdus.writeto(output_name, overwrite=True)
    log.info('Plan initialization saved to {0}'.format(output_name))


if __name__ == '__main__':

    #stop = desisurvey.utils.get_date('2019-10-03')
    #stop = desisurvey.utils.get_date('2020-07-13')
    stop = None
    ephem = desisurvey.ephemerides.Ephemerides(stop_date=stop)
    initialize(ephem, stop_date=stop)