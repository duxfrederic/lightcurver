---
title: '`lightcurver`: A Python pipeline for precise photometry of multiple-epoch wide-field images'
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

`lightcurver` is a photometry pipeline for cadenced astronomical imaging data, 
designed for the semi-automatic extraction of precise light curves from small, blended targets. 
Such targets include, but are not limited to, lensed quasars, supernovae, or cepheids in crowded fields. 
`lightcurver` is not a general-purpose photometry, astrometry, and classification pipeline like `legacypipe` [@legacypipe]. 
Instead, it is a framework tailored for the precise study of a small region of interest (ROI) in wide-field images, 
utilizing stars surrounding the ROI to calibrate the frames.

At its core, `lightcurver` leverages `STARRED` [@starred; @starredscience] to generate state-of-the-art Point Spread Function (PSF) models for each image. 
It then determines the relative zeropoints between images by combining the PSF-photometry fluxes of several stars in the field of view. 
Subsequently, `STARRED` is used again to simultaneously model the calibrated pixels of the ROI across all epochs. 
This process yields light curves of the point sources and a high-resolution image model of the ROI, cumulating the signal from all epochs.

`lightcurver` aims to be maintainable, fast, and incremental in its processing approach. 
As such, it can enable the daily photometric analysis of a large number of blended targets 
in the context of the upcoming Rubin Observatory Legacy Survey of Space and Time (LSST: @LSST)  . 

# Statement of need

The LSST survey will generate an unprecedented amount of imaging data, 
revisiting the same regions of the sky every four days, with irregular pointings due to its observing strategy.
Processing data at this cadence will require robust pipelines capable of ingesting new observations 
and providing immediate photometric calibration and analysis. 
This is particularly important for time-sensitive targets of opportunity, 
where rapid reaction to changes is essential for timely follow-up. 
An existing pipeline that performs this precise deblending and photometric measurement task, `COSMOULINE` [@cosmouline; @MCS], 
requires too much manual intervention to be run on a daily basis.

On the other hand, `STARRED` is a powerful PSF modelling and deconvolution package, ideal for this task. 
However, by its nature, it cannot include an infrastructure that makes it convenient to apply to large datasets without manual intervention
(e.g., visually identifying appropriate stars, extracting cutouts, and all subsequent processing steps leading to a light curve). 
Particularly, `STARRED` modelling requires a very stable zero-point across modelled epochs, 
as it emulates the constant components of the ROI as one grid of pixels common to all epochs, 
which it simultaneously optimizes together with the fluxes of the variables. 
Achieving such precise relative zero-point calibration (typically a milimag), especially in an automated manner, comes with challenges.

`lightcurver` addresses this challenge by automatically selecting calibration stars, modelling them, 
and robustly combining their fluxes to calibrate the zeropoints.
To make it suitable as a daily running pipeline on a large number of ROIs, 
`lightcurver` was designed to be fast, incremental, and capable of automatically reducing new images.


![Light curve of a lensed image of a quasar (J0030-1525), extracted once with the existing code base (`COSMOULINE`), 
requiring a week of investigor's time, and another time with `lightcurver`, requiring about an hour of investigator's time. 
HST image: PI Tommaso Treu, proposal GO 15652.](plot/comparison_with_legacy_pipeline.jpg)


# Functionality

`lightcurver` utilizes an SQLite3 database to track data processing stages and relies on SQL queries to manage its workflow, 
identifying the processing required at each step. 
Firstly, the frames undergo background subtraction, and the sources are extracted using `sep` [@Barbary2016; @sextractor]. 
The positions of the extracted sources are then used to plate-solve each frame, primarily with `Astrometry.net` [@astrometry]. 
This allows for an automatic selection of calibration stars around the region of interest (ROI) by querying Gaia [@gaia] 
with `astroquery` [@astroquery] for suitable stars. 
The pointings and field rotations need not be stable across epochs, as each frame is assigned its own calibration stars.

Subsequently, cutouts of the ROI and stars are extracted using `astropy` [@astropy], masked, 
cleaned from cosmic rays with the help of `astroscrappy` [@astroscrappy; @lacosmic], 
and stored in an HDF5 file [@fortner1998hdf].
The PSF model is then calculated for each frame with `STARRED` before being stored in the same HDF5 file. 
Next, the fluxes of all the calibration stars in all frames are measured using PSF photometry, 
and the resulting fluxes of the stars are scaled and combined to obtain precise relative zeropoints.
To avoid clearly failed fits from spoiling the reduction, 
each fitting procedure (PSF or photometry) stores its reduced chi-squared statistic in the database, 
allowing downstream steps to filter which frame, PSF, or flux it proceeds with. 

The software is divided into three main subpackages: `processes`, containing individual data processing tasks; 
`pipeline`, defining the sequence of these tasks to ensure orderly data analysis; 
and `structure`, which contains the database schema and handles user configuration.
Users can customize the processing of datasets through a YAML configuration file, 
allowing flexibility in handling various data characteristics. 
Typically, the YAML configuration file needs to be configured by the user once when executing 
the pipeline on the first few frames, but the subsequent addition of new frames as 
they are observed requires no further manual intervention.

`lightcurver`, in comparison to `COSMOULINE`, achieves equal or better photometric precision in a much more automated fashion. 
Figure 1 presents the light curve of the northernmost lensed image of the quasar J0030-1525 [@lemon2018], 
extracted from the same dataset (ESO program 0106.A-9005(A), PI Courbin) using both `COSMOULINE` and `lightcurver`. 
The stable zeropoint across frames enables STARRED to reliably fit the constant components, in this case, two galaxies visible in the image. 
This reliable deblending of the different flux components yields both light curves and a high-resolution image, 
whose morphology is confirmed by comparison with Hubble Space Telescope imaging.

In summary, `lightcurver` is a robust and efficient photometry pipeline designed for the semi-automatic extraction 
of precise light curves from small, blended targets in cadenced astronomical imaging data. 
By leveraging the power of STARRED for state-of-the-art PSF modeling and deconvolution, 
and employing an automated flux calibration process, `lightcurver` achieves equal or better photometric precision 
compared to existing pipelines, while requiring significantly less manual intervention.

