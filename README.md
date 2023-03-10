# Synthesizing Sea Surface Temperature and Satellite Altimetry Observations Using Deep Learning Improves the Accuracy and Resolution of Gridded Sea Surface Height Anomalies

Code associated with the paper: [Martin, S., Manucharyan, G., & Klein, P., 2022, Journal of Advances in Modelling Earth Systems](https://doi.org/10.31223/X50Q0N).

![alt text](https://github.com/smartin98/deep-learning-ssh-mapping-JAMES-paper/blob/main/src/NN_architecture.png?raw=true)

## Description

In this study we present a novel deep learning method for more accurately mapping sea surface height (SSH) by combining satellite altimetry and sea surface temperature (SST) observations. 

This repository provides:
* Code to generate training data from the publicly available satellite altimetry and SST observation datasets following the pre-processing steps described in the manuscript
* Code to define, train, and make predictions with the deep learning model described in the manuscript
* Underlying data for the figures presented in the manuscript
* A Jupyter Notebook for generating the results figures in the manuscript
* `.nc` files containing our new SSH maps for the Gulf Stream Extension region
* A `.xlsx` file containing the underlying data for Table A1 in the paper

## Dependencies

* All code is in Python (3.9.5)
* The following packages were used:
    * NumPy (1.21.6)
    * SciPy (1.8.0)
    * TensorFlow (2.6.1)
    * TensorFlow-Addons (0.14.0)
    * Xarray (2022.3.0)
    * Pyproj (3.3.1)
    * Matplotlib (3.5.2)
    * Joblib (1.1.0)
    * Cmocean (2.0)
    * Seaborn (0.11.2)
    * Pandas (1.4.2)
* Scripts for generating training data assume the user has locally downloaded the publicly available satellite altimetry and SST observations to their local machine.

## Structure of repo & instructions

* The `src` folder contains data for the figures and re-used python functions
* The `figures.ipynb` notebook can be run to generate the figures from the paper using the data stored in this repo
* The `make_data.py` script is to create training data assuming you have local copies of the satellite observations with the same naming conventions we used:
    * Level 3 satellite altimetry data downloaded from CMEMS (DOI:10.48670/moi-00146) and stored in a directory with structure: `l3 sla data/satellite_identifier_code/standard_CMEMS_sla_filename.nc`, where the satellite identifier codes are e.g. c2, c2n, j2, etc. as used by CMEMS
    * Level 4 DUACS SSH data downloaded from CMEMS (DOI:10.48670/moi-00148) with directory structure: `duacs/duacs_YYYY-MM-DD.nc`
    * Level 4 GHRSST SST data downloaded from NASA PODAAC (DOI:10.5067/GHGMR-4FJ04) with directory structure: `sst high res/standard_PODAAC_SST_filename.nc`
* The `train_predict.py` script defines our ConvLSTM model, trains the model (assuming you have training data generated from `make_data.py`), and makes prediction on validation data
* Our method's predicted SSH maps are in the `data challenge maps` directory, the maps for other maps can be found in the [ocean data challenge](https://github.com/ocean-data-challenges/2021a_SSH_mapping_OSE) repo

## Authors

* Code by [Scott Martin](https://www.ocean.washington.edu/home/Scott_Martin)
* Paper by Scott Martin, Georgy Manucharyan, & Patrice Klein

Contact: `smart1n@uw.edu`

## Acknowledgments

The results presented in the paper used data and code from this [ocean data challenge](https://github.com/ocean-data-challenges/2021a_SSH_mapping_OSE), we thank the creators and maintainers of this repository for this valuable resource! We have slightly adapted some of their code for our purposes and reproduced it here to ensure the repo is self contained, files with the form `mod_XXX.py` are adapted from the ocean data challenge repo. 
