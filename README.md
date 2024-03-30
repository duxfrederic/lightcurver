<img src="docs/mkdocs/contents/lightcurver_logo.svg" alt="logo" style="width:30em;"/>

[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/lightcurver)](https://pypi.org/project/lightcurver/)
![tests](https://github.com/duxfrederic/lightcurver/actions/workflows/python-app.yml/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/badge/Docs-Available-green)](https://duxfrederic.github.io/lightcurver/)


# LightCurver 

Welcome to `lightcurver`! 
This is a photometry library leveraging [STARRED](https://gitlab.com/cosmograil/starred), 
best used with time series of wide-field images. You can read more about it in the [documentation](https://duxfrederic.github.io/lightcurver/).

`lightcurver` essentially prepares a Point Spread Function (PSF) model for each wide-field image, before using it
to precisely calibrate the relative zero point between each image.
Finally, `STARRED` models the pixels of the region of interest, 
yielding of course high quality light curves of the point sources in the region of interest, 
but also recovering the subpixel information to provide a high signal-to-noise ratio deconvolution of the region of interest itself.
The example below shows a cutout of a wide-field image (one in a set of a hundred), 
the model/deconvolution, and the Hubble Space Telescope image of the same region.

![example_deconvolution](docs/mkdocs/contents/example_deconv.png)

## Features
* Uses plate solving (https://astrometry.net/) to keep track of the footprint of each frame, allowing for an independent selection of reference stars in each frame.
* Leverages _Gaia_ information to select the right reference stars in the field of view.
* Never interpolates: essential to preserve the sub-pixel information that can be reocovered by `STARRED` in a multi-epoch deconvolution.
* Provides an extremely precise relative flux calibration between epochs and a PSF model for each epoch.
* Uses `sqlite3` queries to dynamically determine which process needs be executed on which frame. (adding a new frame does not require the reprocessing of everything).
* Attempts to keep the number of created files to a minimum, this is crucial when working on servers with high lattency storage.


## Getting Started

1. **Installation**: the short version, install via `pip`:

    ```
    pip install lightcurver
    ```
[The slightly longer version](https://duxfrederic.github.io/lightcurver/installation/), in case you plan on using a GPU or the plate solving.

2. **Tutorial** follow the [tutorial](https://duxfrederic.github.io/lightcurver/tutorial/) of the documentation, which provides a dataset you can experiment with.

## The implemented processing steps
![flowdiagram](docs/flow_diagram/workflow_diagram.svg)


## Contributing

Whether you're fixing a bug, implementing a new feature, or improving documentation, your efforts are highly appreciated. 
If you are using this code and are encountering issues, feel free to contact me or to directly open an issue or pull request.

## License

LightCurver is licensed under the GPL v3.0 License. See the [LICENSE](LICENSE) file for more details.

## Contact

Have questions or suggestions? Feel free to open an issue, a pull request, or [reach out](mailto:frederic.dux@epfl.ch)!
