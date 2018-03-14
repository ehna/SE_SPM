# SE_SPM
MATLAB post-processing codes for sequential excitation scanning probe microscopy SE-SPM

This package is used to post-process the sequintial excitation data from Asylum Research AFMs. 
To perform analysis, open and run PCA_SHO_sPFM.m in MATLAB. 
This code imports the IBW files from AR software to matlab, performs image registration to compensate the drift within the SE data set, performs PCA and data-reduction on the data set, and finally performs Simple Harmonic Oscillator fittings on the truncated data set using both CPU and GPU. 

An example of 41 single-frequency PFM images are located in sPFM data Folder.

