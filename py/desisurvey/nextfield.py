#- Provided by Data Systems to be called by DOS

def get_next_field(dateobs, skylevel, seeing, transparency, previoustiles,
    programname=None):
    """
    Returns structure with information about next field to observe.
    
    Args:
        dateobs (float): start time of observation in UTC (TAI).
            Could be past, present, or future.
        skylevel: current sky level [counts/s/cm^2/arcsec^2]
        seeing: current astmospheric seeing PSF FWHM [arcsec]
        transparency: current atmospheric transparency
        previoustiles: list of tile IDs previously observed.
        programname (string, optional): if given, the output result will be for
            that program.  Otherwise, next_field_selector() chooses the
            program based upon the current conditions.

    Returns dictionary with keys
        tileid: tile ID [integer]
            --> DOS should just add this to the raw data header
        programname: DESI (or other) program name, e.g. "Dark Time Survey",
            "Bright Galaxy Survey", etc.
            --> DOS should just add this to the raw data header
        telera, teledec: telescope central pointing RA, dec [J2000 degrees]
        exptime: expected exposure time [seconds]
        maxtime: maximum allowable exposure time [seconds]
        fibers: dictionary with the following keys, each of which contains
            a list of 5000 values for each of the positioners
            - ra: RA for each fiber [J2000 degrees]
            - dec: dec for each fiber [J2000 degrees]
            - lambdaref: wavelength to optimize each positioner [Angstrom]
        gfa: dictionary with the following keys, each of which contains
            a list of values for objects detectecable by the GFAs, including
            border regions in RA,dec to assist with acquisition
            - id : ID of GFA for this object
            - ra, dec : RA and dec for each object [J2000 degrees]
            - objtype : 'point', 'extended', 'sky'
                --> point sources with okguide=True can be used for guiding;
                    knowledge of the existence of extended sources may help
                    with acquisition; sky locations are large enough to be
                    used for estimating sky backgrounds.
            - okguide : True if good for guiding
            - mag : magnitude [SDSS r-band AB magnitude]
                --> or a flux instead?

        Additional keys may be present and should be ignored
        
    e.g. result['fibers']['ra'] gives the 5000 RA locations for the fibers
    
    Notes:
      * get_next_field() will calculate the LST and moon phase/location
        based upon the input datetime.
      * skylevel, seeing, and transparency are in the filter of the guider.
      * The contents of the returned dictionary should be *everything* needed
        as input to point the telescope and take an exposure.  If that isn't
        true, we need to add more.  An ancillary/test/commissioning program
        that defines all of these quantities (e.g. in a JSON file)
        should be sufficient to take observations.
      * previoustiles is a required input rather than having get_next_field()
        query ObsDB to get the history.  Two reasons:
        - Easier to test without requiring live database
        - Decouples code dependencies
      * result['fibers'] will be pre-calculated by fiber assignment;
        DOS shouldn't care as long as get_next_field is fast (<1 sec).
      * Current expectation is the ObsDB/DOS only tracks the past, i.e. what
        observations were taken, but not which observations we would like to
        take in the future.  As such, get_next_field() will need to look up
        the DESI tiling (currently in desimodel/data/footprint/desi-tiles.*)
        and a list of overrides for tiles that were observed by deemed bad
        and need to be redone (details TBD).
        DOS shouldn't care about those details.
        
    TBD:
      * Error handling: if the request is impossible (e.g. the sun is up),
        should this raise an exception?  Or return a default zenith answer
        with some calib programname?  Or?
    """
    raise NotImplementedError
