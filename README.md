# MTM-SVD
Repository for Python implementation of MTM-SVD method of spatiotemporal signal detection and reconstruction.

# Contents
- CESM_LME: folder with necessary MTM and MTM-SVD analysis performed for publication "Multidecadal Temperature Variability in the Community Earth System Model Last Millennium Ensemble". Fernandez et al. (2024)


## Notes on MTM-SVD analysis 

Code based on Mathilde Jutras' python implementation of Correa-Ramirez and Hormazabal 2012 *** MATLAB code.
https://github.com/mathildejutras/mtm-svd-python

Method developed by Mann and Park (1994) ****

1. Download CESM LME data from https://www.cesm.ucar.edu/community-projects/lme
2. Run mtmsvd_preprocessing.py routing the path to your dowloaded data
3. mtmsvd.py calculates the LFV spectra for each simulation of each case in the ensemble. It also calculates the confidence intervals. Route the path to which you wish to save the results
4. mtmsvd_evolving.py calculates the evolving LFV spectrum.
5. mtmsvd_var_exp.py calculates the signal to noise maps for any configuration of ensemble or single simulation of your choosing
6. plot_**.py plots any of the above analyses

## Notes on conventional MTM analysis and wavelet spectra

Single timeseries calculated and analyzed thanks to Pyleoclim package **

## Dependencies

To use Pyleoclim, make sure to have a Python version >3.11. 

## License

The project is licensed under the GNU Public License. Please refer to the file call license. If you use the code in publications, please credit the work using the citation file.

## References

  **  Khider, Deborah, Emile-Geay, Julien, Zhu, Feng, James, Alexander, Landers, Jordan, Kwan, Myron, & Athreya, Pratheek. (2022). Pyleoclim: A Python package for the analysis and visualization of paleoclimate data (v0.9.1). Zenodo. https://doi.org/10.5281/zenodo.7523617

  *** Correa-Ramirez, M., & Hormazabal, S. (2012). MultiTaper Method-Singular Value Decomposition (MTM-SVD): spatial-frequency variability of the sea level in the southeastern Pacific. Latin American Journal of Aquatic Research, 40(4), 1039-1060.


  **** Mann, M. E., & Park, J. (1994). Global‐scale modes of surface temperature variability on interannual to century timescales. Journal of Geophysical Research: Atmospheres, 99(D12), 25819-25833.
