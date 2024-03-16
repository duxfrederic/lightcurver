---
title: 'LightCurver: A Python pipeline for precise photometry of multiple-epoch wide-field images'
tags:
  - Python
  - astronomy
  - pipeline
  - PSF photometry
authors:
  - name: Frédéric Dux
    orcid: 0000-0003-3358-4834
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: European Southern Observatory, Alonso de Córdova 3107, Vitacura, Santiago, Chile
   index: 1
 - name: Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland
   index: 2
date: 18 March 2024
bibliography: paper.bib
---

# Summary

`lightcurver` is a pipeline designed for precise photometry of cadenced astronomical imaging data,
designed for the extraction of precise light curves in small, blended targets, such as lensed quasars, supernovae, or cepheids in crowded fields.
It is thereby not a general purpose photometry, astrometry and classification pipeline like, for example, the `legacypipe` [@legacypipe].
Rather, it is a framework to precisely study small regions of wide-field images, using stars surrounding the region of interest (ROI) to calibrate the frames.
`lightcurver` will be useful in handling the massive quantities of wide-field images expected from the upcoming Rubin Observatory Legacy Survey of Space and Time LSST [@LSST], 
which will periodically revisit the same regions of the sky with irregular pointings due to its observing strategy.
At its core, `lightcurver` leverages `STARRED` [@starred] to generate state-of-the-art Point Spead Function (PSF) models for each frame.
The relative zeropoints between frames are then calculated by combining the PSF-photometry fluxes of several stars in the field of view.
Finally, `STARRED` is used again to simultaneously model the pixels of the region of interest for all epochs at once.

# Statement of need

The LSST survey will generate an unprecedented amount of imaging data, 
requiring robust pipelines capable of ingesting new observations and providing immediate photometric calibration and analysis. 
This is particularly crucial for time-sensitive targets of opportunity, where rapid reaction to changes is essential. 
Moreover, the required photometric precision for many scientific goals is very high, 
necessitating exquisite calibration of the zero-point of each image before attempting photometry. 
This is especially important for blended targets with constant components (the lens galaxy and lensed rings in lensed quasars,
or the host galaxy for supernovae). Such components need be modelled with a constant model across epochs to remove 
the flux degeneracies between the target variables and the constants.
To make it suitable as a daily running pipeline on a large number of ROIs, `lightcurver` was designed to be fast, precise, and able to automatically reduce new data.

# Functionality

`lightcurver` uses an SQLite3 database to track data processing stages and relies on SQL queries to manage its workflow, 
identifying the processing required at each step. 
The potential stars are extracted with `sep` [@Barbary2016, @sextractor], and their positions are used to plate solve 
each frame with `Astrometry.net` [@astrometry] or other alternative strategies.
This then allows for an automatic selection of calibration stars around the ROI by querying Gaia [@gaia] with `astroquery` [@astroquery] for suitable stars.
Cutouts of the ROI and stars are then extracted using `astropy` [@astropy], masked, cleaned from cosmics with the help of `astroscrappy` [@astroscrappy, @lacosmic] and stored in an HDF5 file [@fortner1998hdf].
At every step, the database is used to check which calibration stars are available in which frames.
The PSF model is then calculated for each frame with `STARRED`, before being stored in the same HDF5 file.
Then, the fluxes of all the calibration stars in all frames is measured with PSF photometry. 
The relative fluxes of the stars are scaled and combined to calculate the relative zeropoint of each frame.
The software architecture is divided into two main subpackages: processes, which contains individual data processing tasks, 
and pipeline, which defines the sequence of these tasks to ensure orderly data analysis. 
Users can customize the processing of datasets through a YAML configuration file, allowing for flexibility in the handling of various data characteristics. 
Typically, the YAML configuration file needs be configured by the user once for the first few frames, 
but the subsequent addition of new frames as they are observed should require no further user action.

`lightcurver` is made to be fast, and in comparison to other codes performing the same task, achieves better photometric precision in a much more automated way.
We provide in the figure below the light curve of a lensed image of a quasar, extracted from the same dataset once using `COSMOULINE` [@cosmouline].


![Light curve of a lensed image of a quasar (J0659+1629), extracted once with the existing code base (COSMOULINE), requiring a week of investigor's time, and another time with `LightCurver`, requiring about two hours of investigator's time.](plot/comparison_with_legacy_pipeline.jpg)
