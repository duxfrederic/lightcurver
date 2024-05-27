import numpy as np
import ephem
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u


def calculate_airmass(altitude_degrees: np.array) -> np.array:
    """
    Calculate the airmass using Rozenberg's empirical relation.

    Parameters
    ----------
    altitude_degrees : float or ndarray
        The apparent altitude of the object in degrees. Can be a single value or a NumPy array.

    Returns
    -------
    float or ndarray
        The airmass value(s). Returns -1.0 for altitudes below the horizon (altitude_radians < 0),
        -2.0 for altitudes above 90 degrees (altitude_radians > pi/2), and the airmass according
        to Rozenberg's relation for altitudes in between.

    Notes
    -----
    Rozenberg's empirical relation is defined as:
        X = 1 / (sin(ho) + 0.025 * exp(-11 * sin(ho)))
    where ho is the apparent altitude of the object. This formula is applicable down
    to the horizon (where it gives X = 40).
    """
    altitude_radians = np.radians(np.asarray(altitude_degrees))
    with np.errstate(divide='ignore', invalid='ignore'):
        airmass_values = np.where(
            (altitude_radians < 0),
            -1.0,
            np.where(
                (altitude_radians > np.pi / 2),
                -2.0,
                1.0 / (np.sin(altitude_radians) + 0.025 * np.exp(-11.0 * np.sin(altitude_radians)))
            )
        )
    return airmass_values


def ephemeris(mjd: float,
              ra_object: float, dec_object: float,
              telescope_longitude: float, telescope_latitude: float, telescope_elevation: float) -> dict:
    """
    This function calculates and returns the ephemeris for a given object at a specific time and location.

    Parameters:
    mjd (float): Modified Julian Date for the observation.
    ra_object (float): Right ascension of the observed object in degrees.
    dec_object (float): Declination of the observed object in degrees.
    telescope_longitude (float): Longitude of the telescope's location in degrees.
    telescope_latitude (float): Latitude of the telescope's location in degrees.
    telescope_elevation (float): Elevation of the telescope's location in meters.

    Returns:
    dict: A dictionary containing information about the astro conditions, comments, target, moon, and sun.
        - 'astro_conditions' (bool): Indicates if the astro conditions are favorable.
        - 'comments' (str): Additional comments about the astro conditions.
        - 'target_info' (dict): Information about the target object, including altitude, azimuth, and airmass.
        - 'moon_info' (dict): Information about the moon, including distance from the target, illumination, and altitude.
        - 'sun_info' (dict): Information about the sun, including altitude.
    """

    results = {
        'weird_astro_conditions': False,
        'comments': "",
        'target_info': {},
        'moon_info': {},
        'sun_info': {}
    }

    telescope = ephem.Observer()
    # Convert longitude and latitude to strings
    long_str = Angle(telescope_longitude * u.deg).to_string(unit=u.degree, sep=':')
    lat_str = Angle(telescope_latitude * u.deg).to_string(unit=u.degree, sep=':')
    telescope.long = long_str
    telescope.lat = lat_str
    telescope.elevation = telescope_elevation
    telescope.epoch = ephem.J2000

    jd = mjd + 2400000.5
    djd = jd - 2415020.0
    telescope.date = djd

    target = ephem.FixedBody()
    # make them coordinate strings for pyephem:
    coord = SkyCoord(ra=ra_object*u.degree, dec=dec_object*u.degree)
    ra_str = coord.ra.to_string(unit=u.hour, sep=':')
    dec_str = coord.dec.to_string(unit=u.degree, sep=':')
    target._ra = ra_str
    target._dec = dec_str
    target.compute(telescope)

    moon = ephem.Moon()
    moon.compute(telescope)
    moondist = np.degrees(float(ephem.separation(moon, target)))

    sun = ephem.Sun()
    sun.compute(telescope)

    target_alt_deg = np.degrees(target.alt)
    target_az_deg = np.degrees(target.az)

    airmass = calculate_airmass(target_alt_deg)
    if airmass < 1.0 or airmass > 5.0:
        results['weird_astro_conditions'] = True
        results['comments'] += f"Target altitude: {target_alt_deg:.2f} degrees (airmass {airmass:.2f})."

    moon_distance_deg = np.degrees(ephem.separation(moon, target))
    moon_illumination = moon.phase
    moon_alt_deg = np.degrees(moon.alt)

    sun_alt_deg = np.degrees(sun.alt)
    if sun_alt_deg > 0.0:
        results['weird_astro_conditions'] = True
        results['comments'] += f" Sun altitude: {sun_alt_deg:.2f} degrees."

    # Fill target, moon, and sun info
    results['target_info'] = {'altitude_deg': target_alt_deg,
                              'azimuth_deg': target_az_deg,
                              'airmass': airmass,
                              'moon_dist': moondist}
    results['moon_info'] = {'distance_deg': moon_distance_deg,
                            'illumination': moon_illumination,
                            'altitude_deg': moon_alt_deg}
    results['sun_info'] = {'altitude_deg': sun_alt_deg}

    return results


def estimate_seeing(sources_table):
    """
    logic written by Malte Tewes, https://github.com/mtewes in 2010, for the COSMOULINE pipe of COSMOGRAIL.
    it seems to have worked well for 14 years now, so I'm just keeping the main flow of it.

    this function estimates a seeing value (in pixels) based on a table of sources as extracted by sep.
    we need the FWHM column to be present in the table, hence the table of sources has to come from
    the extract_stars function of this repo.

    :param sources_table: astropy.table.Table containing detections
    :return: float, a seeing value (reasonable FWHM) in pixels.
    """

    fwhms = sources_table['FWHM']

    if len(fwhms) > 10:  # loads of stars
        # We want a crude guess at what kind of range we have to look for stars
        # The goal here is to have a "nice-looking"
        # histogram with a well-defined peak somewhere inside the range.
        min_fwhm = 1.5
        med_fwhm = np.median(fwhms)
        if med_fwhm < min_fwhm:
            med_fwhm = min_fwhm

        wide_stars = 3.0 * med_fwhm

        max_fwhm = 30.0
        if wide_stars < max_fwhm:
            max_fwhm = wide_stars

        # At this point the true seeing should be between min_fwhm and max_fwhm.
        # We build a first histogram :
        (hist, edges) = np.histogram(fwhms, bins=10,
                                     range=(min_fwhm, max_fwhm))
        # Note that points outside the range are not taken into account at all,
        # they don't fill the side bins!

        # We find the peak, and build a narrower hist around it
        max_pos = np.argmax(hist)
        if max_pos == 0:
            seeing_pixels = np.median(fwhms)
        elif max_pos == len(hist) - 1:
            seeing_pixels = np.median(fwhms)
        else:  # the normal situation :
            peak_pos = float(0.5 * (edges[max_pos] + edges[max_pos + 1]))

            # We build a second histogram around this position,
            # with a narrower range:
            hist, edges = np.histogram(fwhms,
                                       bins=10,
                                       range=(peak_pos - 2.0, peak_pos + 2.0))
            max_pos = np.argmax(hist)
            peak_pos = 0.5 * (edges[max_pos] + edges[max_pos + 1])

            # We take the median of values around this peak_pos :
            star_fwhms = fwhms[np.logical_and(fwhms > peak_pos - 1.0,
                                              fwhms < peak_pos + 1.0)]
            if len(star_fwhms) > 0:
                seeing_pixels = np.median(star_fwhms)
            else:
                seeing_pixels = peak_pos

    elif len(fwhms) > 0:  # few stars, not ideal but whatever
        seeing_pixels = np.median(fwhms)

    else:  # no stars
        seeing_pixels = -1.0
    return seeing_pixels
