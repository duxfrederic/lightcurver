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

`lightcurver` is a photometry pipeline of cadenced astronomical imaging data,
designed for the semi-automatic extraction of precise light curves in small, blended targets.
Such targets include but are not limited to lensed quasars, supernovae, or cepheids in crowded fields.
`lightcurver` is thereby not a general purpose photometry, astrometry and classification pipeline such as the `legacypipe` [@legacypipe].
Rather, it is a framework made for the precise study a small region of interest (ROI) in wide-field images, 
using stars surrounding the ROI to calibrate the frames.
At its core, `lightcurver` leverages `STARRED` [@starred] to generate state-of-the-art Point Spead Function (PSF) models for each image.
It then determines the relative zeropoints between images by combining the PSF-photometry fluxes of several stars in the field of view.
Finally, `STARRED` is used again to simultaneously model the calibrated pixels of the ROI of all epochs at once.
This process yields light curves of the point sources, and a high resolution image model of the ROI cumulating the signal of all epochs.
`lightcurver` aims at being maintainable and fast, and as such will hopefully allow the daily photometric analysis of 
a large number of blended targets in the context of the upcoming Rubin Observatory Legacy Survey of Space and Time LSST [@LSST].
LSST will revisit the same regions of the sky every four days, with irregular pointings due to its observing strategy.


# Statement of need

The LSST survey will generate an unprecedented amount of imaging data, 
requiring robust pipelines capable of ingesting new observations and providing immediate photometric calibration and analysis. 
This is particularly true for time-sensitive targets of opportunity, where rapid reaction to changes is essential for timely follow-up.
An existing pipeline which does this precise task, `COSMOULINE` [@cosmouline], requires too much manual intervention
to be run on a daily basis.
On the other hand, `STARRED` is a powerful PSF modelling and deconvolution package and is ideal for this task,
but it by nature cannot include an infrastructure that makes it convenient to apply to large datasets without manual intervention
(e.g., visually identifying the appropriate stars, extracting cutouts, and all the subsequent processing steps that lead to a light curve).
In particular, `STARRED` modelling requires a very stable zero-point across modelled epochs, for it models
the constant components of the ROI as one pixel grid common to all epochs, which it simultaneously optimizes together with the 
fluxes of the variables. 
Such a precise relative zero-point calibration comes with challenges, especially if it needs be automated.
`lightcurver` addresses this challenge by automatically selecting calibration stars, modelling them, and combining
their fluxes to calibrate the zeropoints in a robust way.
To make it suitable as a daily running pipeline on a large number of ROIs, 
`lightcurver` was designed to be fast, precise, and able to reduce new images automatically.


# Functionality

`lightcurver` uses an SQLite3 database to track data processing stages and relies on SQL queries to manage its workflow, 
identifying the processing required at each step. 
The potential stars are extracted with `sep` [@Barbary2016, @sextractor], and their positions serve to plate solve 
each frame with `Astrometry.net` [@astrometry] or other alternative strategies.
This then allows for an automatic selection of calibration stars around the ROI by querying Gaia [@gaia] with `astroquery` [@astroquery] for suitable stars.
Cutouts of the ROI and stars are subsequently extracted using `astropy` [@astropy], masked, 
cleaned from cosmics with the help of `astroscrappy` [@astroscrappy, @lacosmic] and stored in an HDF5 file [@fortner1998hdf].
At every step, the database is used to check which calibration stars are available in which frames.
The PSF model is then calculated for each frame with `STARRED`, before being stored in the same HDF5 file.
Next, the fluxes of all the calibration stars in all frames is measured with PSF photometry, 
and the resulting fluxes of the stars are scaled and combined to obtain precise relative zeropoints.
To avoid clearly failed fits spoiling the reduction, each fitting procedure (PSF or photometry) stores its reduced $\chi^2$ statistic in
the database, allowing the downstream steps to filter which frame / PSF / flux it moves forward with. 
The software architecture is divided into three main subpackages: `processes`, which contains individual data processing tasks, 
`pipeline`, which defines the sequence of these tasks to ensure orderly data analysis,
and `structure`, which contains the database schema and handles the user configuration.
Users can customize the processing of datasets through a YAML configuration file, 
allowing for flexibility in the handling of various data characteristics. 
Typically, the YAML configuration file needs be configured by the user once when executing the pipeline on the first few frames, 
but the subsequent addition of new frames as they are observed should require no further manual intervention.

`lightcurver` is made to be fast, and in comparison to `COSMOULINE`, achieves equal or better photometric precision, in a much more automated way.
We provide in the figure below the light curve of a lensed image of a quasar, extracted from the same dataset using both `COSMOULINE` and `lightcurver`.


![Light curve of a lensed image of a quasar (J0659+1629), extracted once with the existing code base (COSMOULINE), requiring a week of investigor's time, and another time with `LightCurver`, requiring about an hour of investigator's time. HST image: PI Tommaso Treu, proposal GO 15652](plot/comparison_with_legacy_pipeline.jpg)
