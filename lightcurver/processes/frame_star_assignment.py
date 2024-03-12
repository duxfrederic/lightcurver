from shapely.geometry import Point, Polygon
from shapely import intersection
import json
import sqlite3
import numpy as np

from ..structure.user_config import get_user_config


def populate_stars_in_frames():
    """
    Populates the stars_in_frames table by checking which stars fall within each frame's footprint.
    """

    user_config = get_user_config()
    # not using execute_sqlite_query, so we do not open and close the database on million times.
    conn = sqlite3.connect(user_config['database_path'])
    cursor = conn.cursor()

    # load the footprints and stars
    cursor.execute("""SELECT frame_id, polygon FROM footprints""")
    frame_footprints = cursor.fetchall()
    cursor.execute("""SELECT gaia_id, ra, dec, combined_footprint_hash FROM stars""")
    stars = cursor.fetchall()

    for frame_id, footprint_str in frame_footprints:
        # assume gnomonic projection, let's just treat our footprint as flat.
        # should be good enough.
        footprint_polygon = Polygon(json.loads(footprint_str))
        # we don't want to select stars too close to the edges, too many problems.
        # to address this, we'll shrink our polygon by a margin.

        # get the edges:
        x, y = footprint_polygon.exterior.xy
        # the mean declination:
        mean_dec = np.nanmean(y)
        margin_degrees = 4. / 3600  # enforce a margin of 15 arcseconds.
        # just de-projecting the margin for RA:
        ra_margin = margin_degrees / np.cos(np.radians(mean_dec))

        # I do not know how to shrink y polygon by different amounts in different directions ...
        # no worries, we'll just translate the polygon by the amount in a cross-like pattern,
        # then take the intersection to get our 'shrunk' polygon.
        translated_polygons = []
        for translation in ([1, 0], [-1, 0], [0, -1], [0, 1]):
            adjusted_vertices = []
            for ra, dec in zip(x, y):
                adjusted_ra = ra + translation[0] * ra_margin
                adjusted_dec = dec + translation[1] * margin_degrees
                adjusted_vertices.append((adjusted_ra, adjusted_dec))
            translated_polygons.append(Polygon(adjusted_vertices))

        # do the intersection of our translations to get the reduced footprint:
        shrunk_footprint_polygon = translated_polygons[0]
        for trans in translated_polygons[1:]:
            shrunk_footprint_polygon = intersection(shrunk_footprint_polygon, trans)

        # and now we check!
        for star_id, ra, dec, combined_footprint_hash in stars:
            star_point = Point(ra, dec)
            if star_point.within(shrunk_footprint_polygon):
                try:
                    cursor.execute("""INSERT INTO stars_in_frames (frame_id, star_gaia_id, combined_footprint_hash) 
                                      VALUES (?, ?, ?)""", (frame_id, star_id, combined_footprint_hash))
                except sqlite3.IntegrityError:
                    # handles cases where the same star-frame relationship might already be in the table
                    continue

    conn.commit()
    conn.close()

