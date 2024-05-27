def parse_header(header):
    from dateutil import parser
    from astropy.time import Time
    exptime = header['exptime']
    gain = header['gain']
    filter = header['filter']
    time = Time(parser.parse(header['obstart']))
    return {'exptime': exptime, 'gain': gain, 'filter': filter, 'mjd': time.mjd}