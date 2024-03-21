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


So, I would suggest creating an Anaconda environment `lightcurver_env` and install `STARRED` in it.
Next, install `astrometry.net` on your system. (Do not forget to download index files, their combined 4200 and 5100 series should work for most use cases).
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
The automated tests also include the processing of a subset of the dataset given in the tutorial, you can thus run them
instead to check functionality. 
```bash
cd /your/clone/of/lightcurver
pytest .
```
