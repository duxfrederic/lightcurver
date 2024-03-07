import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from graphviz import Digraph

dot = Digraph()

#dot.node('A', 'Initialize Database\nJust making sure the database exists, with the right schema.', shape='box')
dot.node('B', 'Read/Convert/SkySub/Character Catalog\nreads the images from a directory, converts them to electrons,\nsubtracts the sky, finds sources, measures noise, seeing,\ncalculates ephemerides, records all to database', shape='box')
dot.node('C', 'Plate Solving\nUses astrometry.net\'s solve-field and initial guess of plate scale\nprovided in user config to add a reliable WCS to each image fast.\nIf user config states that images are already plate solved, skip.', shape='box')
dot.node('D', 'Calculate Common and Total Footprint\nChecks the footprint of the images, see how big the common footprint is.\n', shape='box')
dot.node('E', 'Query Gaia for Stars\nGiven the footprints above, finds stars in gaia for PSF modelling and normalization.\n', shape='box')
dot.node('F', 'Stamp Extraction\nExtracts stamps of all good stars and all epochs.\nAlso extract stamps of the region of interest (ROI).\nSaves the stamps to an HDF5 file\n Also cleans the cosmics.', shape='box')
dot.node('G', 'PSF Modeling\nCreates a PSF model for each frame', shape='box')
dot.node('H', 'Star Photometry\nUses the PSF model to do PSF photometry of each star, using STARRED\n(joint deconvolution). The fluxes (per frame and per star) are saved', shape='box')
dot.node('I', 'Calculate Normalization Coefficient\nGiven the star photometry, calculates a representative relative flux for each image.\n', shape='box')
dot.node('J', 'Prepare Calibrated Cutouts\nPrepares cutouts for each ROI and each frame, calibrated in flux by the normalization coefficient.', shape='box')
dot.node('K', 'Deconvolution\nThe user can now run STARRED on the prepared cutouts.', shape='box')

dot.edges(['BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK'])

dot.render('workflow_diagram', format='svg', cleanup=True)

