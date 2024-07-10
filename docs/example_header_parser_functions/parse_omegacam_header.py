def parse_header(header):
    from dateutil import parser
    from astropy.time import Time
    exptime = header['exptime']
    gain = header['gain']
    time = Time(parser.parse(header['obstart']))
    return {'exptime': exptime, 'gain': gain, 'mjd': time.mjd}