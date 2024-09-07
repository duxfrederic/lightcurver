---
title: Installation
weight_index: 9
---
# Installation
`lightcurver` requires several components to function:

- `STARRED`: it will be installed automatically with `pip install lightcurver`, but if you want to use it with a GPU there
might be some more setup to do. See the [installation instructions](https://cosmograil.gitlab.io/starred/installation.html#) of the package itself.
- The dependencies handled by your python package manager, such as `astropy`, `shapely`, `astroquery`, `pandas`, `pyyaml` 
...these will be installed automatically by `pip install lightcurver`.
- (optional) `Astrometry.net`: their [installation instructions](https://astrometry.net/doc/build.html) should get you started. 
Alternatively, you can get an API key from their [nova](https://nova.astrometry.net/) service. I would recommend against using it in production, as to not overload their servers.


So, I would suggest creating a python (3.9+, ideally 3.11) environment, say `lightcurver_env`,
and install the present package in it.

## The quick version
Chances are this will work:
```bash
    conda activate lightcurver_env # if using conda
    source lightcurver_env/bin/activate # if using python's venv
    pip install lightcurver
```

Or for the `git` version (includes some minimal test data):
```bash
    git clone git@github.com:duxfrederic/lightcurver.git
    cd lightcurver
    conda activate lightcurver_env
    pip install .
```

## If the quick version fails: list of dependencies
Should the above fail, there might be a dependency problem requiring the manual handling of the different packages. 
Here is the list of dependencies that need be installed:

1. `numpy < 2.00` - as of June 2024, `sep` is not compatible with `numpy >= 2.00`
2. `scipy`
3. `matplotlib`
4. `pandas`
5. `astropy`
6. `astroquery` - for querying Gaia and VizieR
7. `h5py` - for storing cutouts and PSF models
8. `photutils` - for aperture photometry used as initial guess
9. `astroalign` - for finding transformations between frames
10. `shapely` - for computing footprints of frames
11. `ephem` - for calculating airmass, moon distance, etc.
12. `pytest` - for executing the automatic tests
13. `sep` - for source and background extraction
14. `astroscrappy` - for cleaning the cosmics
15. `pyyaml` - for reading the config file
16. `starred-astro` - assume the latest version, will install its own dependencies.
17. `widefield_plate_solver` - an astrometry.net wrapper


## Testing your installation

You can test your installation by following the [tutorial](tutorial.md).
The automated tests also include the processing of a subset of the dataset given in the tutorial, you can thus run them
instead to check functionality (should take 1-2 minutes). 
```bash
cd /your/clone/of/lightcurver
pytest .
```

If you are going to use a local installation of `Astrometry.net`, do not forget to download their index files as well! The combination of 4100 and 5200 should do the trick.
