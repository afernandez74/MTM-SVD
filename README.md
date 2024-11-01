# MTM-SVD
Repository to house all things related to mtm-svd analysis of climate data

# Contents
- mtm-svd functions (lfv, conf intervals, reconstruction and envel)
- calculate annual means function
- python scripts that run a variety of experiments/analyses

## Instructions

1. Download CESM LME data from https://www.cesm.ucar.edu/community-projects/lme
2. Run mtmsvd_preprocessing.py routing the path to your dowloaded data
3. mtmsvd.py calculates the LFV spectra for each simulation of each case in the ensemble. It also calculates the confidence intervals. Route the path to which you wish to save the results
4. mtmsvd_evolving.py calculates the evolving LFV spectrum.
5. mtmsvd_var_exp.py calculates the signal to noise maps for any configuration of ensemble or single simulation of your choosing
6. plot_**.py plots any of the above analyses
