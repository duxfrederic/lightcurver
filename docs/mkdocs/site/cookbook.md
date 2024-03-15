---
title: Cookbook
weight: 7
---
# Cookbook and random fixes

`LightCurver` will absolutely fail you a lot. Sorry, astronomical data is just too messy and such is life.
Here I will add example situations and how to fix them.

## Some of my images were imported, but cannot be plate solved due to low quality
This is a classic. High airmass observations, clouds, tracking problems ...
If you imported such images and are confident that you will not be able to extract value from them, you can 
simply execute them with a database operation:
````sqlite3
'UPDATE frames SET comment='cannot be plate solved', eliminated = 1 where PLATE_SOLVED=0;'
````
## I manually plate solved my images after importation, how can I make my pipeline aware of this?
You need to manually run the process storing the footprints in the database and checking that your region of interest
is in the frame.
Here is how you might do that, with your current directory set to your working directory.
```python
import os
os.environ['LIGHTCURVER_CONFIG'] = "/path/to/config.yaml"
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

from lightcurver.processes.plate_solving import post_plate_solve_steps
from lightcurver.structure.user_config import get_user_config
from lightcurver.structure.database import execute_sqlite_query

user_config = get_user_config()

solved = Path('frames').glob('*.fits')

for s in solved:
    s = str(s)
    if 'sources.fits' in s:
        # this is a table of sources, skip
        continue
    wcs = WCS(fits.getheader(s))
    if not wcs.is_celestial:
        # this one wasn't solved then
        continue
    frame_id = execute_sqlite_query('select id from frames where image_relpath = ?', 
                                    params=(s,), is_select=True)[0][0]
    
    try:
       post_plate_solve_steps(frame_path=s, user_config=user_config, frame_id=frame_id)
    except AssertionError:
         # already inserted
        pass

    execute_sqlite_query(query="UPDATE frames SET plate_solved = ? WHERE id = ?",
                         params=(1, frame_id), is_select=False)
```
