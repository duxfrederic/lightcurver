---
title: Installation
weight_index: 9
---
# Installation
`LightCurver` requires several components to function:

- `STARRED`: it will be installed automatically with `pip install lightcurver`, but if you want to use it with a GPU there
might be some more setup to do. See the [installation instructions](https://cosmograil.gitlab.io/starred/installation.html#) of the package itself.
- The dependencies handled by your python package manager, such as `astropy`, `shapely`, `astroquery`, `pandas`, `pyyaml` 
...these will be installed automatically by `pip install lightcurver`.
- (optional) `Astrometry.net`: their [installation instructions](https://astrometry.net/doc/build.html) should get you started. 
Alternatively, you can get an API key from their [nova](https://nova.astrometry.net/) service. I would recommend against using it in production, as to not overload their servers.


So, I would suggest creating a python (3.9+, ideally 3.11) environment, say `lightcurver_env`,
and install the present package in it:

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

You can test your installation by following the [tutorial](tutorial.md).
The automated tests also include the processing of a subset of the dataset given in the tutorial, you can thus run them
instead to check functionality (should take 1-2 minutes). 
```bash
cd /your/clone/of/lightcurver
pytest .
```

If you are going to use a local installation of `Astrometry.net`, do not forget to download their index files as well! The combination of 4100 and 5200 should do the trick.
