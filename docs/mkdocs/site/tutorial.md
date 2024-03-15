# LightCurver tutorial

## Introduction
By default, `lightcurver` is a pipeline which executes all its steps sequentially, 
through the `lightcurver.pipeline.workflow_manager.WorkflowManager` class.
So, the most basic python script executing the pipeline would be as follows:

```python
import os
os.environ['LIGHTCURVER_CONFIG'] = "/path/to/config.yaml"

from lightcurver.pipeline.workflow_manager import WorkflowManager
wf_manager = WorkflowManager()
wf_manager.run()
```

Where the `config.yaml` file needs to be carefully tuned before execution. You should always start from 
[this template](https://github.com/duxfrederic/lightcurver/blob/main/docs/example_config_file/config.yaml).

Rather than executing the pipeline through the `WorkflowManager`, we will first execute each step manually in this tutorial.

I will provide you with some wide field images that you can use to follow along. Note the following:

* the images are already plate solved, such that you will not need to install `Astrometry.net` on your computer.
Your real life examples will most likely not be, so you should consider installing it.
* Things will be excruciatingly slow if you do not have a GPU. I would consider using only 4-5 frames in this case.

You can work in a jupyter notebook or just write python scripts with the commands we will execute below.
## Preparing the working directory and data

The example dataset consists of a few frames from the monitoring of a lensed quasar with the VLT survey telescope.
You can find it at this [link](https://www.astro.unige.ch/~dux/vst_dataset_example.zip), but we will download it below anyway.
Start by creating a working directory. I will assume the working directory `/scratch/lightcurver_tutorial`, please
replace this with your own.
```bash
mkdir /scratch/lightcurver_tutorial
```
Let us store our raw data in this working directory as well for the sake of the example, but of course it could be
anywhere, including on a network drive.
```bash
cd /scratch/lightcurver_tutorial
wget https://www.astro.unige.ch/~dux/vst_dataset_example.zip
unzip vst_dataset_example.zip
```
Your data will now be at `/scratch/lightcurver_tutorial/raw`.

The pipeline also expects a function able to read the header of your fits files. Store the following python function:
```python
def parse_header(header):
    from dateutil import parser
    from astropy.time import Time
    exptime = header['exptime']
    gain = header['gain']
    filter = header['filter']
    time = Time(parser.parse(header['obstart']))
    return {'exptime': exptime, 'gain': gain, 'filter': filter, 'mjd': time.mjd}
```
in this file: `/scratch/lightcurver_tutorial/header_parser/parse_header.py`. 
The pipeline expects to find this file at this exact location relative to your working directory.
You will need to adapt the function to your own fits files, the point is: you must return a dictionary of the same
structure as the one seen above. You can of course use placeholder values should you not care, for example, about
the filter information. 
> **__NOTE:__** We will use the `exptime` and `gain` information to convert the images to electrons per second, assuming that the starting unit is ADU. Please adapt the values you return within this function should your units be different.

Now, we need to set up the configuration file of the pipeline. This file could be anywhere, but we will put it in
our working directory. 
I provide a [fairly generic configuration](https://github.com/duxfrederic/lightcurver/blob/main/docs/example_config_file/config.yaml) 
which works well for this particular dataset.
Paste the contents of the file in `/scratch/lightcurver_tutorial/config.yaml`.
You will most probably need to adapt these lines at least:
```yaml
workdir: /scratch/lightcurver_tutorial 
# ...
raw_dirs:
  - /scratch/lightcurver_tutorial/raw
# further below ...
already_plate_solved: 1
```
This last line informs the pipeline about the plate solved status of our files.
You can also read through the configuration file to learn about the different options of the pipeline.

At this point, you could just run the code block at the very beginning of this page, and the pipeline would likely
run to the end, producing an `hdf5` file with calibrated cutouts and PSFs of our region of interest.
However, we will execute each step separately, so you get a chance to look at the outputs.

## Initializing database and frame importation
Now would be a good time to fire up a jupyter notebook, each code block below being a new cell.
You first need to add the location of your config file to the environment, then you can start executing tasks:
```python
import os
# replace with your actual path:
os.environ['LIGHTCURVER_CONFIG'] = "/scratch/lightcurver_tutorial/config.yaml"

from lightcurver.structure.user_config import get_user_config
from lightcurver.structure.database import initialize_database
from lightcurver.pipeline.task_wrappers import read_convert_skysub_character_catalog

initialize_database()
read_convert_skysub_character_catalog()
```

This last command will read all the frames, convert them to electron / second (we are assuming ADU as initial units),
subtract the sky, look for sources in the image, calculate ephemeris and finally store everything in our database, at
`/scratch/lightcurver_tutorial/database.sqlite3`.
> **_NOTE:_** You may query the database at any time to understand what is going on.
>For example, at the moment we have:
>>`$ sqlite3 /scratch/lightcurver_tutorial/database.sqlite3 "select count(*) from frames"`
>>`87`
> 
>87 frames imported. The database contains the frames, and later will contain stars, links between stars and frames, and more.

## Plate solving and footprint calculation
Even though we started with plate solved images, we are still going to call the plate solving routine.
No actual plate solving will take place, but the footprint of each image will be inserted in the database,
and we will calculate the total and common footprint to all images. This can be useful if you want to make sure
that you are always going to use the same reference stars, in each frame. 
Let us go ahead and run the task:
```python
# assuming the path to the config file is still in the environment
from lightcurver.pipeline.task_wrappers import plate_solve_all_frames, calc_common_and_total_footprint_and_save
plate_solve_all_frames()
calc_common_and_total_footprint_and_save()
```
This will have populated the `footprints`, `combined_footprint`.

> **_NOTE:_** All downstream steps from this one are linked to a hash value of the combined footprint.
> This is due to the fact that the reference stars can be queried in the common footprint: adding a new frame
> would potentially change the common footprint, and thus, the stars.

## Querying stars from Gaia

In the configuration, I recommend using
```yaml
star_selection_strategy: 'ROI_disk'
ROI_disk_radius_arcseconds: 300  # think twice about this value: it has to contain enough stars, but not too many
```
Next, depending on your data, you will need to adjust the accepted magnitude range (to include stars that are
bright enough, but that do not saturate the sensor.)
If you have good seeing and are working with oversampled data, I recommend sticking to a relatively low value of
`star_max_astrometric_excess_noise`. Gaia can sometimes mistake a galaxy for a star, and a galaxy would do no
good to your PSF model. Keeping the astrometric excess noise low (e.g., below 3-4) largely reduces the risk
of selecting a galaxy.
This is how this part is executed:
```python
# assuming the path to the config file is still in the environment
from lightcurver.processes.star_querying import query_gaia_stars
query_gaia_stars()
```
This will populate the `stars` and `stars_in_frames` tables of the database. The latter allows us to query
which star is available in which frame.

## Extraction of cutouts
Now that we've identified stars, let us extract them from the image. This step will
- extract the cutouts
- compute a noisemap (from the background noise, and photon noise estimation given that we can convert our data to electrons)
- clean the cosmics (unless stated otherwise in the config)
and that for each selected star, and also for our region of interest.
These will all go into the `regions.h5` file, at the root of the working directory.

## Modelling the PSF
This is the most expensive step of the pipeline. For each frame, we are going to simultaneously fit a
grid of pixels to all the selected stars. The grid of pixels being regulated by starlets, we delegate the heavy 
lifting to `STARRED`.
I recommend sticking to a subsampling factor of 2 unless you have good reasons to go beyond.
You can expect the process to last 2-3 seconds per frame on a middle range gaming GPU, including the loading of the data,
the modelling, the plotting, and database update.

```python
# assuming the path to the config file is still in the environment
from lightcurver.processes.psf_modelling import model_all_psfs
model_all_psfs()
```

This will populate the `PSFs` table in the database, saving the subsampling factor, the reduced $\chi^2$ of the fit,
and a string reminding which stars were used to compute the model.

## PSF photometry of the reference stars
This step will, for each star
- select which frames contain this star
- eliminate frames with a poorly fit PSF (looking at the $\chi^2$ values, check the config file for how this is done)
- jointly fit the PSF to the star in question in all the selected frames.

```python
# assuming the path to the config file is still in the environment
from lightcurver.processes.star_photometry import do_star_photometry
do_star_photometry()
```
The fitted fluxes will be saved in the `star_flux_in_frame` table, together with, again, a $\chi^2$ value that
will be used downwstream to eliminate the badly fitted frames.

## Calculating a normalization coefficient
This step leverages all the extracted star fluxes, and scales them as to minimize the scatter of the fluxes of
different stars in overlaping frames.
Once this is done, the fluxes available in each frame will be averaged with sigma-clipping rejection.
The average will be taken as the "normalization coefficient", and the residual scatter as the uncertainty on the
coefficient.

```python
# assuming the path to the config file is still in the environment
from lightcurver.processes.normalization_calculation import calculate_coefficient
calculate_coefficient()
```
This process fills in the `normalization_coefficients` table.

## Calculating zero points and preparing calibrated cutouts of our region of interest
All the heavy lifting having been done, we can use our Gaia stars to estimate the absolute zero point of our images.
This is an approximate calibration, but it is nice to have. 
Then, we will use our normalization coefficient to prepare the calibrated cutouts.
```python
# assuming the path to the config file is still in the environment
from lightcurver.utilities.zeropoint_from_gaia import calculate_zeropoints
from lightcurver.processes.roi_deconv_file_preparation import prepare_roi_deconv_file
calculate_zeropoints()
prepare_roi_deconv_file()
```
You will find your calibrated cutouts in the `prepared_roi_cutouts`, relative to the working directory