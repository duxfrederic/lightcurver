from shapely.geometry import Point, Polygon
import json
import sqlite3

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

        for star_id, ra, dec, combined_footprint_hash in stars:
            star_point = Point(ra, dec)
            if star_point.within(footprint_polygon):
                try:
                    cursor.execute("""INSERT INTO stars_in_frames (frame_id, gaia_id, combined_footprint_hash) 
                                      VALUES (?, ?, ?)""", (frame_id, star_id, combined_footprint_hash))
                except sqlite3.IntegrityError:
                    # Handle cases where the same star-frame relationship might already be in the table
                    continue

    conn.commit()
    conn.close()

