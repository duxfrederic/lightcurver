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

