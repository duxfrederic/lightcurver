![LightCurver Logo](docs/mkdocs/site/lightcurver_logo.svg)

![tests](https://github.com/duxfrederic/lightcurver/actions/workflows/python-app.yml/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# LightCurver 

Welcome to `lightcurver`! 
This is a photometry library leveraging [STARRED](https://gitlab.com/cosmograil/starred), 
best used with time series of wide-field images. 
`lightcurver` essentially prepares a Point Spread Function (PSF) model for each wide-field image, before using it
to precisely calibrate the relative zero point between each image.
Finally, `STARRED` models the pixels of the region of interest, 
yielding of course high quality light curves of the point sources in the region of interest, 
but also recovering the subpixel information to provide a high signal-to-noise ratio deconvolution of the region of interest itself.
The example below shows a cutout of a wide-field image (one in a set of a hundred), 
the `lightcurver` / `STARRED` model, and the Hubble Space Telescope image of the same region.

![example_deconvolution](docs/mkdocs/site/example_deconv.png)

## Features
* Uses plate solving to keep track of the footprint of each frame, allowing for an independent selection of reference stars in each frame.
* Leverages _Gaia_ information to select the right reference stars in the field of view.
* Never interpolates: essential to preserve the sub-pixel information that can be reocovered by `STARRED` in a multi-epoch deconvolution.
* Provides an extremely precise relative flux calibration between epochs and a state-of-the-art PSF model for each epoch.
* Uses `sqlite3` queries to dynamically determine which process needs be executed on which frame. (adding a new frame does not require the reprocessing of everything).
* Attempts to keep the number of created files to a minimum, this is crucial when working on servers with high lattency storage.


## Getting Started

0. **Requirements**: on top of the python libraries listed in `requirements.txt`, we need either
    - a working installation of `astrometry.net`, which provides the `solve-field` function.
    - alternatively, an `astrometry.net` API key.

1. **Installation**: Clone the repository and install via `pip`:

    ```
    git clone git@github.com:duxfrederic/lightcurver.git
    cd lightcurver
    pip install -e lightcurver
    ```

2. **Usage**:
There are several preparation steps to complete before you can start analyzing your wide field images.
- Define a working directory, we will call it `workdir`. 
- Create a subdirectory, `header_parser`, in `workdir`, and create a python file: `$workdir/header_parser/parse_header.py`.  This file should contain a function, `parse_header`, which should extract the exposure time, the gain, and the MJD or some other time information. See the [example header parser](docs/example_header_parser_functions/) directory for an example.
- Copy the [example config file](docs/example_config_file/config.yaml), and update it with your information.
Now you can run `lightcurver`:
    ```python
    # important before importing: tell the code where your config file is
    import os
    os.environ['LIGHTCURVER_CONFIG'] = "/path/to/your_config.yaml"

    from lightcurver.pipeline.workflow_manager import WorkflowManager
    wf_manager = WorkflowManager()
    wf_manager.run()
    ```
`lightcurver` will run several steps of analysis on your wide-field images, like a pipeline.

## The implemented processing steps
![flowdiagram](docs/flow_diagram/workflow_diagram.svg)

## Roadmap 
- Finish the logging system: we should be left with one unified logging interface.
- refactor some of the tasks, ideally all functions that do database queries should go in `processes`.

## Contributing

Whether you're fixing a bug, implementing a new feature, or improving documentation, your efforts are highly appreciated. 
If you are using this code and are encountering issues, feel free to contact me or to directly open an issue or pull request.

## License

LightCurver is licensed under the GPL v3.0 License. See the [LICENSE](LICENSE) file for more details.

## Contact

Have questions or suggestions? Feel free to open an issue, a pull request, or [reach out](mailto:frederic.dux@epfl.ch)!
