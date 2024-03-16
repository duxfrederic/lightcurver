---
title: Installation
weight_index: 9
---
# Installation
`LightCurver` requires several components to function:

- `STARRED`: see the [installation instructions](https://cosmograil.gitlab.io/starred/installation.html#) of the package itself. A GPU is strongly recommended, but you can of course run it on CPU -- just arm yourself with patience.
- (optional) `Astrometry.net`: the installation instructions should get you started. Alternatively, you can get an API key from their [nova](https://nova.astrometry.net/) service. I would recommend against using it in production, as to not overload their servers.
- The dependencies handled by your python package manager, such as `astropy`, `shapely`, `astroquery`, `pandas`, `pyyaml` ...these will be installed automatically.

So, I would suggest creating an Anaconda environment `lightcurver_env` and install `STARRED` in it.
Next, install `astrometry.net` on your system. (Do not forget to download index files, their 4200 and 5200 series should work).
Finally, install the present package:

```bash
    conda activate lightcurver_env
    pip install lightcurver
```

Or for the latest version:
```bash
    git clone git@github.com:duxfrederic/lightcurver.git
    cd lightcurver
    conda activate lightcurver_env
    pip install .
```


You can test your installation by following the [tutorial](tutorial.md).

